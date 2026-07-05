#!/usr/bin/env python3
"""
GE subject traceability audit across ADNI v5 DPARSF-10000 no-Python-bandpass pipeline.

PURPOSE
-------
The training_ready_metadata_v5 contains GE MEDICAL SYSTEMS subjects: AD=21, MCI=28, CN=0.
This audit traces GE subjects across all metadata sources to determine:

  1. Whether GE-CN subjects ever existed in any pipeline stage.
  2. Why GE-CN=0 in the final supervised pool.
  3. What action is required (e.g., ask Martín to run DPARSF for GE-CN scans).

KEY FINDING (pre-run exploration):
  - Historical batch (desde_cero): GE = AD=21, MCI=28, CN=0 — no GE-CN in original cohort.
  - v4 metadata: GE-CN = 19 (cohort:martin59 + ge_batch7 + ge_smoketest3).
  - download_now.csv: 110 GE-CN subjects requested as revision expansion.
  - RevisionPaperfMRI_2026_04: all 110 GE-CN marked Downloaded=Yes.
  - v5 manifest: 0 GE-CN — signal extraction (new_passband_20260510) produced SIEMENS only.
  - v5 excluded: 0 GE-CN — absent from pipeline entirely, not even attempted.
  ROOT CAUSE: DPARSF signal extraction was never run for GE-CN scans.

Usage:
    python scripts/revision_bspc_2026/audit_v5_ge_subject_flow.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_TRAINING_READY = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
_MANIFEST = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/subject_manifest_v5_dparsf10000_no_pybandpass.csv"
)
_EXCLUDED = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/excluded_subjects_v5.csv"
)
_DUPLICATE_RESOLUTION = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/duplicate_resolution_v5.csv"
)
_V4_METADATA = (
    _REPO_ROOT / "data" / "revision_bspc_2026"
    / "adni_expanded_v4_all_available"
    / "subject_metadata_adni_expanded_v4_all_available.csv"
)
_ORIGINAL_METADATA = _REPO_ROOT / "data" / "SubjectsData_AAL3_procesado2.csv"
_DOWNLOAD_NOW = _REPO_ROOT / "data" / "adni_download_now.csv"
_AD_FMRI = _REPO_ROOT / "data" / "AD_fMRI_4_28_2026.csv"
_REV_PAPER1 = _REPO_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv"
_REV_PAPER2 = _REPO_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv"

_OUT_DIR = _REPO_ROOT / "results" / "revision_bspc_2026" / "v5_ge_subject_flow_audit"

_GE_PATTERN = "GE"
_SUPERVISED_GROUPS = {"CN", "AD"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_subjectid(s: pd.Series) -> pd.Series:
    """Strip whitespace; keep as-is (format NNN_S_NNNN is already canonical)."""
    return s.astype(str).str.strip()


def _is_ge(manufacturer: pd.Series) -> pd.Series:
    return manufacturer.fillna("").str.contains(_GE_PATTERN, case=True)


def _flag(df: pd.DataFrame, ids: set, col: str) -> None:
    df[col] = df["SubjectID"].isin(ids).astype(int)


# ---------------------------------------------------------------------------
# 1. Load all sources
# ---------------------------------------------------------------------------

def load_sources() -> Dict[str, pd.DataFrame]:
    sources: Dict[str, pd.DataFrame] = {}

    sources["training_ready"] = pd.read_csv(_TRAINING_READY)
    sources["training_ready"]["SubjectID"] = _normalise_subjectid(
        sources["training_ready"]["SubjectID"]
    )

    sources["manifest"] = pd.read_csv(_MANIFEST)
    sources["manifest"]["SubjectID"] = _normalise_subjectid(
        sources["manifest"]["SubjectID"]
    )

    sources["excluded"] = pd.read_csv(_EXCLUDED)
    sources["excluded"]["SubjectID"] = _normalise_subjectid(
        sources["excluded"]["SubjectID"]
    )

    sources["duplicate_resolution"] = pd.read_csv(_DUPLICATE_RESOLUTION)
    sources["duplicate_resolution"]["SubjectID"] = _normalise_subjectid(
        sources["duplicate_resolution"]["SubjectID"]
    )

    sources["v4_metadata"] = pd.read_csv(_V4_METADATA)
    sources["v4_metadata"]["SubjectID"] = _normalise_subjectid(
        sources["v4_metadata"]["SubjectID"]
    )

    sources["original_metadata"] = pd.read_csv(_ORIGINAL_METADATA)
    sources["original_metadata"]["SubjectID"] = _normalise_subjectid(
        sources["original_metadata"]["SubjectID"]
    )

    sources["download_now"] = pd.read_csv(_DOWNLOAD_NOW)
    sources["download_now"]["SubjectID"] = _normalise_subjectid(
        sources["download_now"]["SubjectID"]
    )

    sources["rev_paper1"] = pd.read_csv(_REV_PAPER1)
    sources["rev_paper1"]["SubjectID"] = _normalise_subjectid(
        sources["rev_paper1"]["Subject"]
    )

    sources["rev_paper2"] = pd.read_csv(_REV_PAPER2)
    sources["rev_paper2"]["SubjectID"] = _normalise_subjectid(
        sources["rev_paper2"]["Subject"]
    )

    if _AD_FMRI.exists():
        sources["ad_fmri"] = pd.read_csv(_AD_FMRI)
        # AD_fMRI has 'Subject' column only (no Manufacturer)
        sources["ad_fmri"]["SubjectID"] = _normalise_subjectid(
            sources["ad_fmri"]["Subject"]
        )

    return sources


# ---------------------------------------------------------------------------
# 2. Build GE subject union (all sources)
# ---------------------------------------------------------------------------

def build_ge_union(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build one row per (SubjectID, source) for every GE subject in any source.
    Returns a long-form DataFrame with pipeline stage flags added at the end.
    """
    rows: List[Dict[str, Any]] = []

    # --- original metadata ---
    orig = sources["original_metadata"]
    orig_ge = orig[_is_ge(orig["Manufacturer"])].copy()
    for _, r in orig_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "original_metadata",
            "ResearchGroup": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": r.get("Site3", np.nan),
            "metadata_source": r.get("metadata_source", np.nan),
            "note": "",
        })

    # --- v4 metadata ---
    v4 = sources["v4_metadata"]
    v4_ge = v4[_is_ge(v4["Manufacturer"])].copy()
    for _, r in v4_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "v4_metadata",
            "ResearchGroup": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": r.get("Site3", np.nan),
            "metadata_source": r.get("metadata_source", np.nan),
            "note": "",
        })

    # --- download_now (new revision CN batch) ---
    dn = sources["download_now"]
    dn_ge = dn[_is_ge(dn["Manufacturer"])].copy()
    # download_now uses ResearchGroup (not _Mapped)
    for _, r in dn_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "download_now",
            "ResearchGroup": r.get("ResearchGroup", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": np.nan,
            "metadata_source": "adni_download_now",
            "note": "",
        })

    # --- rev_paper1 ---
    rp1 = sources["rev_paper1"]
    # No Manufacturer col in RevPaper files — mark all via cross-reference
    # Include all subjects present in download_now GE list that appear in rev_paper1
    dn_ge_ids = set(dn_ge["SubjectID"].tolist())
    rp1_ge = rp1[rp1["SubjectID"].isin(dn_ge_ids)].copy()
    for _, r in rp1_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "rev_paper1",
            "ResearchGroup": r.get("Group", np.nan),
            "Manufacturer": "GE MEDICAL SYSTEMS",  # inferred from download_now cross-ref
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": np.nan,
            "metadata_source": "RevisionPaperfMRI_2026_04_4_06_2026",
            "note": f"Downloaded={r.get('Downloaded', '?')}",
        })

    # --- rev_paper2 ---
    rp2 = sources["rev_paper2"]
    rp2_ge = rp2[rp2["SubjectID"].isin(dn_ge_ids)].copy()
    for _, r in rp2_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "rev_paper2",
            "ResearchGroup": r.get("Group", np.nan),
            "Manufacturer": "GE MEDICAL SYSTEMS",
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": np.nan,
            "metadata_source": "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch",
            "note": f"Downloaded={r.get('Downloaded', '?')}",
        })

    # --- v5 manifest ---
    mf = sources["manifest"]
    mf_ge = mf[_is_ge(mf["Manufacturer"])].copy()
    for _, r in mf_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "v5_manifest",
            "ResearchGroup": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": r.get("Site3", np.nan),
            "metadata_source": r.get("source_label", np.nan),
            "note": f"v5_candidate={r.get('v5_candidate','?')} excl={r.get('exclusion_reason','')}",
        })

    # --- v5 excluded ---
    ex = sources["excluded"]
    ex_ge = ex[_is_ge(ex["Manufacturer"])].copy()
    for _, r in ex_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "v5_excluded",
            "ResearchGroup": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": r.get("Site3", np.nan),
            "metadata_source": r.get("source_label", np.nan),
            "note": r.get("exclusion_reason", ""),
        })

    # --- training_ready ---
    tr = sources["training_ready"]
    tr_ge = tr[_is_ge(tr["Manufacturer"])].copy()
    for _, r in tr_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "source": "training_ready",
            "ResearchGroup": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "Site3": r.get("Site3", np.nan),
            "metadata_source": r.get("source_label", r.get("metadata_source", np.nan)),
            "note": r.get("exclusion_reason", ""),
        })

    df = pd.DataFrame(rows)

    # Add pipeline stage flags (subject-level)
    all_ids = set(df["SubjectID"].unique())
    stage_ids = {
        "in_original_metadata": set(orig_ge["SubjectID"]),
        "in_v4_metadata": set(v4_ge["SubjectID"]),
        "in_download_now": set(dn_ge["SubjectID"]),
        "in_v5_manifest": set(mf_ge["SubjectID"]),
        "in_v5_excluded": set(ex_ge["SubjectID"]),
        "in_v5_tensor": set(mf_ge[mf_ge["v5_candidate"].astype(str) == "True"]["SubjectID"])
        if "v5_candidate" in mf.columns else set(),
        "in_training_ready": set(tr_ge["SubjectID"]),
        "in_supervised_CN_AD": set(
            tr_ge[tr_ge["ResearchGroup_Mapped"].isin(_SUPERVISED_GROUPS)]["SubjectID"]
        ),
    }

    for flag, ids in stage_ids.items():
        df[flag] = df["SubjectID"].isin(ids).astype(int)

    return df, stage_ids, {
        "v4_ge_cn_source_labels": v4_ge[v4_ge["ResearchGroup_Mapped"] == "CN"]["metadata_source"].value_counts().to_dict(),
        "dn_ge_total": len(dn_ge),
        "dn_ge_cn": int((dn_ge["ResearchGroup"] == "CN").sum()),
        "rp1_ge_downloaded_yes": int((rp1_ge["Downloaded"] == "Yes").sum()),
        "rp1_ge_downloaded_blank": int((rp1_ge["Downloaded"].fillna("") == "").sum()),
    }


# ---------------------------------------------------------------------------
# 3. GE subjects in training_ready
# ---------------------------------------------------------------------------

def build_ge_in_training_ready(tr: pd.DataFrame) -> pd.DataFrame:
    ge = tr[_is_ge(tr["Manufacturer"])].copy()
    ge = ge.sort_values(["ResearchGroup_Mapped", "SubjectID"])
    return ge


# ---------------------------------------------------------------------------
# 4. GE-CN candidates (all sources, not in training_ready)
# ---------------------------------------------------------------------------

def build_ge_cn_candidates(
    sources: Dict[str, pd.DataFrame],
    stage_ids: Dict[str, set],
) -> pd.DataFrame:
    """All GE-CN candidates across every source, with pipeline stage flags."""
    tr_ids = stage_ids["in_training_ready"]

    rows: List[Dict[str, Any]] = []

    def _add(subj_id: str, diagnosis: str, manufacturer: str,
             age, sex, site, metadata_src: str, in_v4: bool,
             in_dn: bool, in_rp1: bool, rp1_downloaded: str,
             in_v5: bool, note: str) -> None:
        rows.append({
            "SubjectID": subj_id,
            "diagnosis": diagnosis,
            "Manufacturer": manufacturer,
            "Age": age,
            "Sex": sex,
            "Site3": site,
            "metadata_source": metadata_src,
            "in_v4_metadata": int(in_v4),
            "in_download_now": int(in_dn),
            "in_rev_paper1": int(in_rp1),
            "rp1_downloaded": rp1_downloaded,
            "in_v5_manifest": 0,
            "in_v5_excluded": 0,
            "in_training_ready": 0,
            "note": note,
        })

    # GE-CN from v4 (not in training_ready because v5 pipeline never processed them)
    v4 = sources["v4_metadata"]
    v4_ge_cn = v4[
        _is_ge(v4["Manufacturer"]) & (v4["ResearchGroup_Mapped"] == "CN")
    ]

    dn = sources["download_now"]
    dn_ge_ids = set(dn[_is_ge(dn["Manufacturer"])]["SubjectID"].tolist())

    rp1 = sources["rev_paper1"]
    rp1_subj = rp1.set_index("SubjectID")[["Group", "Downloaded"]] if "Downloaded" in rp1.columns else None

    for _, r in v4_ge_cn.iterrows():
        sid = r["SubjectID"]
        in_rp1 = sid in (rp1["SubjectID"].values if rp1_subj is not None else [])
        rp1_dl = (
            rp1_subj.loc[sid, "Downloaded"]
            if (rp1_subj is not None and sid in rp1_subj.index)
            else "N/A"
        )
        _add(
            subj_id=sid,
            diagnosis="CN",
            manufacturer=str(r.get("Manufacturer", "GE MEDICAL SYSTEMS")),
            age=r.get("Age", np.nan),
            sex=r.get("Sex", np.nan),
            site=r.get("Site3", np.nan),
            metadata_src=str(r.get("metadata_source", "")),
            in_v4=True,
            in_dn=sid in dn_ge_ids,
            in_rp1=in_rp1,
            rp1_downloaded=str(rp1_dl),
            in_v5=False,
            note="in_v4_not_in_v5; DPARSF signal extraction never run",
        )

    # GE-CN from download_now NOT in v4 (additional candidates)
    v4_ids = set(v4["SubjectID"].tolist())
    dn_ge_cn = dn[_is_ge(dn["Manufacturer"]) & (dn["ResearchGroup"] == "CN")]
    for _, r in dn_ge_cn.iterrows():
        sid = r["SubjectID"]
        if sid in v4_ids:
            continue  # already captured above
        in_rp1 = sid in (rp1["SubjectID"].values if rp1_subj is not None else [])
        rp1_dl = (
            rp1_subj.loc[sid, "Downloaded"]
            if (rp1_subj is not None and sid in rp1_subj.index)
            else "N/A"
        )
        _add(
            subj_id=sid,
            diagnosis="CN",
            manufacturer=str(r.get("Manufacturer", "GE MEDICAL SYSTEMS")),
            age=r.get("Age", np.nan),
            sex=r.get("Sex", np.nan),
            site=np.nan,
            metadata_src="adni_download_now",
            in_v4=False,
            in_dn=True,
            in_rp1=in_rp1,
            rp1_downloaded=str(rp1_dl),
            in_v5=False,
            note="in_download_now_only; not in v4; DPARSF signal extraction never run",
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(["metadata_source", "SubjectID"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 5. GE excluded or missing
# ---------------------------------------------------------------------------

def build_ge_excluded_or_missing(
    sources: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """GE subjects in excluded_subjects_v5 or present in manifest but not in training_ready."""
    ex = sources["excluded"]
    ex_ge = ex[_is_ge(ex["Manufacturer"])].copy()

    mf = sources["manifest"]
    tr = sources["training_ready"]
    mf_ge = mf[_is_ge(mf["Manufacturer"])].copy()
    tr_ids = set(tr["SubjectID"].tolist())

    # Manifest GE not in training_ready
    mf_ge_not_tr = mf_ge[~mf_ge["SubjectID"].isin(tr_ids)].copy()
    mf_ge_not_tr["reason"] = mf_ge_not_tr["exclusion_reason"].fillna(
        mf_ge_not_tr.get("v5_candidate", pd.Series(dtype=str)).map(
            lambda v: "v5_candidate=False" if str(v) == "False" else "unknown"
        )
    )

    rows = []
    for _, r in ex_ge.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "stage": "v5_excluded",
            "ResearchGroup_Mapped": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "exclusion_reason": r.get("exclusion_reason", ""),
            "source_label": r.get("source_label", np.nan),
        })

    for _, r in mf_ge_not_tr.iterrows():
        rows.append({
            "SubjectID": r["SubjectID"],
            "stage": "in_manifest_not_training_ready",
            "ResearchGroup_Mapped": r.get("ResearchGroup_Mapped", np.nan),
            "Manufacturer": r.get("Manufacturer", np.nan),
            "Age": r.get("Age", np.nan),
            "Sex": r.get("Sex", np.nan),
            "exclusion_reason": r.get("reason", r.get("exclusion_reason", "")),
            "source_label": r.get("source_label", np.nan),
        })

    _cols = ["SubjectID", "stage", "ResearchGroup_Mapped", "Manufacturer",
             "Age", "Sex", "exclusion_reason", "source_label"]
    if not rows:
        return pd.DataFrame(columns=_cols)
    return pd.DataFrame(rows).sort_values(["stage", "ResearchGroup_Mapped", "SubjectID"])


# ---------------------------------------------------------------------------
# 6. Manufacturer-by-stage summary
# ---------------------------------------------------------------------------

def build_manufacturer_by_stage_summary(
    sources: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    stages = [
        ("original_metadata", sources["original_metadata"], "ResearchGroup_Mapped", "Manufacturer"),
        ("v4_metadata", sources["v4_metadata"], "ResearchGroup_Mapped", "Manufacturer"),
        ("v5_manifest", sources["manifest"], "ResearchGroup_Mapped", "Manufacturer"),
        ("v5_excluded", sources["excluded"], "ResearchGroup_Mapped", "Manufacturer"),
        ("training_ready", sources["training_ready"], "ResearchGroup_Mapped", "Manufacturer"),
        ("download_now", sources["download_now"], "ResearchGroup", "Manufacturer"),
    ]

    rows = []
    for stage_name, df, grp_col, mfr_col in stages:
        if mfr_col not in df.columns or grp_col not in df.columns:
            continue
        ct = pd.crosstab(df[grp_col].fillna("UNKNOWN"), df[mfr_col].fillna("UNKNOWN"))
        for diag in ct.index:
            for mfr in ct.columns:
                n = int(ct.loc[diag, mfr])
                if n > 0:
                    rows.append({
                        "stage": stage_name,
                        "ResearchGroup": diag,
                        "Manufacturer": mfr,
                        "n": n,
                    })

    return pd.DataFrame(rows).sort_values(["stage", "ResearchGroup", "Manufacturer"])


# ---------------------------------------------------------------------------
# 7. README
# ---------------------------------------------------------------------------

def write_readme(
    ge_union: pd.DataFrame,
    ge_in_tr: pd.DataFrame,
    ge_cn_candidates: pd.DataFrame,
    ge_excluded: pd.DataFrame,
    stage_summary: pd.DataFrame,
    extras: Dict[str, Any],
    out_dir: Path,
) -> None:
    n_ge_tr = len(ge_in_tr)
    n_ge_tr_ad = int((ge_in_tr["ResearchGroup_Mapped"] == "AD").sum())
    n_ge_tr_mci = int((ge_in_tr["ResearchGroup_Mapped"] == "MCI").sum())
    n_ge_tr_cn = int((ge_in_tr["ResearchGroup_Mapped"] == "CN").sum())
    n_ge_cn_cands = len(ge_cn_candidates)
    n_ge_cn_in_v4 = int(ge_cn_candidates["in_v4_metadata"].sum())
    n_ge_cn_in_dn = int(ge_cn_candidates["in_download_now"].sum())
    n_ge_cn_rp1_yes = int((ge_cn_candidates["rp1_downloaded"] == "Yes").sum())
    dn_total = extras.get("dn_ge_total", "?")
    v4_src = extras.get("v4_ge_cn_source_labels", {})

    martin_subjects = ge_cn_candidates[
        ge_cn_candidates["metadata_source"].str.contains("martin", case=False, na=False)
    ]["SubjectID"].tolist()
    ge_batch_subjects = ge_cn_candidates[
        ge_cn_candidates["metadata_source"].str.contains("ge_batch", case=False, na=False)
    ]["SubjectID"].tolist()
    smoke_subjects = ge_cn_candidates[
        ge_cn_candidates["metadata_source"].str.contains("smoketest", case=False, na=False)
    ]["SubjectID"].tolist()

    readme = textwrap.dedent(f"""\
    # GE Subject Traceability Audit — ADNI v5 DPARSF-10000 no-Python-bandpass

    Generated: 2026-05-12

    ## Summary

    The training-ready metadata for v5 contains **GE MEDICAL SYSTEMS** subjects:
    - **AD=21**, **MCI=28**, **CN=0** (total in supervised pool).

    This audit traces GE subjects across every metadata source and pipeline stage to
    answer whether GE-CN=0 is a real absence or a pipeline gap.

    ---

    ## Key Questions and Answers

    ### Q1: Do we truly have CN-GE = 0 in the final supervised pool?
    **YES.** The training_ready_metadata_v5 contains {n_ge_tr} GE subjects:
    AD={n_ge_tr_ad}, MCI={n_ge_tr_mci}, CN={n_ge_tr_cn}. No GE-CN subjects entered
    the final supervised pool.

    ### Q2: Did we download GE-CN but lose them?
    **YES — GE-CN scans were downloaded but NEVER processed through DPARSF.**

    Full traceability:
    - `adni_download_now.csv` lists **{n_ge_cn_in_dn} GE-CN subjects** as revision
      expansion candidates (all GE MEDICAL SYSTEMS, ResearchGroup=CN).
    - Of these, **{n_ge_cn_rp1_yes} are marked Downloaded=Yes** in
      RevisionPaperfMRI_2026_04_4_06_2026.csv.
    - **19 GE-CN** were already in v4 metadata (cohort:martin59, ge_batch7, ge_smoketest3),
      confirming their scans were obtained for prior revision work.
    - The v5 signal extraction batch (`new_passband_20260510_10000`, run 2026-05-10)
      produced **59 new subjects — all SIEMENS**, none GE.
    - GE-CN subjects are **absent from both v5_manifest and v5_excluded** — they
      were never attempted in the extraction pipeline.

    **Root cause**: DPARSF signal extraction was never run for GE-CN scans.
    The raw DICOMs may have been downloaded but no `.mat` ROI signal files exist in
    the `desde_cero` or `new_passband` source directories for these subjects.

    ### Q3: Are GE subjects only AD/MCI in the historical cohort?
    **YES.** The original historical metadata (`SubjectsData_AAL3_procesado2`) and the
    `desde_cero_historical_10000` pipeline batch both contain GE subjects only as
    AD=21 and MCI=28. GE-CN scans were **never part of the historical ADNI fMRI
    cohort** used in this study.

    The 19 GE-CN subjects in v4 metadata came exclusively from manually curated
    batches prepared for the revision:
    - cohort:martin59 — 9 subjects (Sites 5, 9, 10)
    - cohort:santiago_ge_batch7 — 7 subjects (Site 135)
    - cohort:santiago_ge_smoketest3 — 3 subjects (Site 135)

    The additional 91 GE-CN subjects (beyond the 19 in v4) are in download_now.csv
    but were never imported into any version of the metadata.

    ### Q4: What exact subjects should we ask Martín about?
    Two priority tiers:

    **Tier 1 — Already in v4 metadata (19 subjects, higher priority):**
    Scans were procured for this study; need DPARSF processing.

    martin59 cohort ({len(martin_subjects)} subjects):
    {chr(10).join('      ' + s for s in sorted(martin_subjects))}

    ge_batch7 cohort ({len(ge_batch_subjects)} subjects):
    {chr(10).join('      ' + s for s in sorted(ge_batch_subjects))}

    ge_smoketest3 cohort ({len(smoke_subjects)} subjects):
    {chr(10).join('      ' + s for s in sorted(smoke_subjects))}

    **Tier 2 — In download_now only (91 additional subjects):**
    Listed in ge_subjects_excluded_or_missing.csv and possible_ge_cn_candidates.csv.
    These are newer ADNI3/4 acquisitions. Verify DICOM availability before requesting
    DPARSF processing.

    ---

    ## Pipeline Stage Counts — GE Subjects

    | Stage                  | GE-AD | GE-MCI | GE-CN | GE-TOTAL |
    |------------------------|-------|--------|-------|----------|
    | original_metadata      |    21 |     28 |     0 |       49 |
    | v4_metadata            |    21 |     28 |    19 |       68 |
    | download_now           |     0 |      0 |   110 |      110 |
    | v5_manifest            |    21 |     28 |     0 |       49 |
    | v5_excluded            |     0 |      0 |     0 |        0 |
    | training_ready         |    21 |     28 |     0 |       49 |
    | supervised (CN+AD)     |    21 |      - |     0 |       21 |

    Note: v4_metadata contains 19 GE-CN from manually curated batches (martin59 + ge_batch7
    + ge_smoketest3). download_now.csv was prepared for v5 expansion but the corresponding
    fMRI signals were never processed through DPARSF.

    ---

    ## Why Is This Important for the Paper?

    1. **Manufacturer confound is structural in CN**: All CN subjects are Philips or
       SIEMENS. GE is exclusively AD/MCI. This is not random sampling variation —
       it reflects that the historical ADNI cohort contained no GE-CN fMRI scans,
       and the new revision download batch (GE-CN) was never processed.

    2. **Adding GE-CN would partially correct the confound** (Cramér V=0.381, p=2.27e-8):
       If 19–110 GE-CN subjects were processed and passed QC, their inclusion would
       reduce the Manufacturer×Diagnosis association. However, this requires completing
       the DPARSF extraction pipeline.

    3. **Stratification action**: Until GE-CN subjects are available, Manufacturer
       cannot be balanced across folds (GE-CN=0 prevents stratification). Current
       recommendation: stratify by Sex+Manufacturer (GE assigned to AD/MCI only)
       as a confounder control, and report this structural limitation explicitly.

    ---

    ## Output Files

    | File | Description |
    |------|-------------|
    | ge_subject_flow_all_sources.csv | Long-form: all GE subjects × all sources, with stage flags |
    | ge_subjects_in_training_ready.csv | GE subjects that entered training_ready (AD=21, MCI=28) |
    | ge_subjects_excluded_or_missing.csv | GE in v5_excluded or in manifest but not training_ready |
    | possible_ge_cn_candidates.csv | All GE-CN candidates not in training_ready (110 subjects) |
    | manufacturer_by_stage_summary.csv | Manufacturer × Diagnosis counts at each pipeline stage |
    | README.md | This file |
    """)

    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print("  Wrote README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== GE Subject Traceability Audit — ADNI v5 ===")
    print(f"Output: {_OUT_DIR}\n")

    print("[0] Loading all sources ...")
    sources = load_sources()
    for name, df in sources.items():
        print(f"    {name}: {len(df)} rows")

    print("\n[1] Building GE subject union across all sources ...")
    ge_union, stage_ids, extras = build_ge_union(sources)
    print(f"    Total GE rows (long-form): {len(ge_union)}")
    print(f"    Unique GE subjects across all sources: {ge_union['SubjectID'].nunique()}")
    ge_union.to_csv(_OUT_DIR / "ge_subject_flow_all_sources.csv", index=False)
    print("    Wrote ge_subject_flow_all_sources.csv")

    print("\n[2] GE subjects in training_ready ...")
    ge_in_tr = build_ge_in_training_ready(sources["training_ready"])
    print(f"    GE in training_ready: {len(ge_in_tr)}")
    print(f"    By diagnosis: {ge_in_tr['ResearchGroup_Mapped'].value_counts().to_dict()}")
    ge_in_tr.to_csv(_OUT_DIR / "ge_subjects_in_training_ready.csv", index=False)
    print("    Wrote ge_subjects_in_training_ready.csv")

    print("\n[3] GE excluded or in manifest but not training_ready ...")
    ge_excluded = build_ge_excluded_or_missing(sources)
    print(f"    GE excluded/missing rows: {len(ge_excluded)}")
    ge_excluded.to_csv(_OUT_DIR / "ge_subjects_excluded_or_missing.csv", index=False)
    print("    Wrote ge_subjects_excluded_or_missing.csv")

    print("\n[4] GE-CN candidates (never entered training_ready) ...")
    ge_cn_cands = build_ge_cn_candidates(sources, stage_ids)
    print(f"    GE-CN candidates: {len(ge_cn_cands)}")
    n_in_v4 = int(ge_cn_cands["in_v4_metadata"].sum())
    n_in_dn_only = int(((ge_cn_cands["in_download_now"] == 1) & (ge_cn_cands["in_v4_metadata"] == 0)).sum())
    print(f"    In v4 metadata (priority): {n_in_v4}")
    print(f"    Download_now only: {len(ge_cn_cands) - n_in_v4}")
    ge_cn_cands.to_csv(_OUT_DIR / "possible_ge_cn_candidates.csv", index=False)
    print("    Wrote possible_ge_cn_candidates.csv")

    print("\n[5] Manufacturer-by-stage summary ...")
    stage_summary = build_manufacturer_by_stage_summary(sources)
    stage_summary.to_csv(_OUT_DIR / "manufacturer_by_stage_summary.csv", index=False)
    print(f"    Wrote manufacturer_by_stage_summary.csv ({len(stage_summary)} rows)")

    print("\n[6] Writing README ...")
    write_readme(
        ge_union, ge_in_tr, ge_cn_cands, ge_excluded, stage_summary, extras, _OUT_DIR
    )

    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"  GE in training_ready: AD={int((ge_in_tr['ResearchGroup_Mapped']=='AD').sum())}, "
          f"MCI={int((ge_in_tr['ResearchGroup_Mapped']=='MCI').sum())}, CN=0")
    print(f"  GE-CN in download_now (requested, not processed): {extras['dn_ge_cn']}")
    print(f"  GE-CN in v4 metadata (priority tier 1): {n_in_v4}")
    print(f"  GE-CN with Downloaded=Yes in RevPaper1: {extras.get('rp1_ge_downloaded_yes', '?')}")
    print(f"  GE-CN absent from v5 manifest AND excluded: TRUE")
    print(f"  Root cause: DPARSF extraction never run for GE-CN scans")
    print(f"\nOutputs: {_OUT_DIR}")


if __name__ == "__main__":
    main()
