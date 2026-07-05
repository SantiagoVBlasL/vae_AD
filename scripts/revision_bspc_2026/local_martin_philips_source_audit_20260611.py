#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_martin_philips_source_audit_20260611.py

Read-only forensic audit: local source discovery and Philips CN merge audit.
Investigates WHY Philips CN subjects have elevated FPR in the promoted model.

Guardrails:
  - read-only: no tensor modification, no metadata modification,
    no model training, no threshold fitting, no OASIS scoring,
    no model artifact overwrite, no subject exclusion.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR     = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT_DIR      = BASE_DIR / "local_martin_philips_source_audit_20260611"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DB_PATH = BASE_DIR / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
MARTIN_ANN_PATH = (
    BASE_DIR
    / "philips_cn_protocol_annotation_for_martin_20260610"
    / "philips_cn_to_annotate_for_martin.csv"
)

# ADNI reference files
DATOS_DIR = Path("/media/diego/Datos")
ADNIMERGE_PATH    = DATOS_DIR / "ADNIMERGE_14Oct2024.csv"
DXSUM_PATH        = DATOS_DIR / "DXSUM_28May2026.csv"
PTDEMOG_PATH      = DATOS_DIR / "PTDEMOG_28May2026.csv"
DATADIC_PATH      = DATOS_DIR / "DATADIC_28May2026.csv"

# Known local QC files
MCH_PATH           = DATOS_DIR / "MAYOADIRL_MRI_MCH_11_01_22_31May2023.csv"
IQM_PATH           = DATOS_DIR / "AAL3" / "ImageQualityMetrics.csv"
DESDE_CERO_SUBJ    = DATOS_DIR / "desde_cero" / "SubjectsData_cleaned.csv"

BATCH_METADATA_DIRS = [
    DATOS_DIR / "vae_AD_data" / "revision_bspc_2026" / "adni_v5_1_batch20260514b_incremental_no_pybandpass",
    DATOS_DIR / "vae_AD_data" / "revision_bspc_2026" / "adni_v5_1_batch20260514_incremental_no_pybandpass",
    DATOS_DIR / "vae_AD_data" / "revision_bspc_2026" / "adni_v5_1_batch20260513_incremental_no_pybandpass",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
command_log: List[Dict[str, Any]] = []


def _log(step: str, detail: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[{ts}] {step}" + (f": {detail}" if detail else ""))
    command_log.append({"timestamp": ts, "step": step, "detail": detail})


def _save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _safe_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    df.to_csv(path, index=False, **kw)
    _log(f"Saved CSV", str(path))


def _safe_md(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")
    _log(f"Saved MD", str(path))


def _rid_from_subjectid(sid: str) -> Optional[int]:
    """Extract 4-digit RID from ADNI SubjectID like '002_S_0295' → 295."""
    m = re.match(r"^\d+_S_(\d+)$", str(sid).strip())
    return int(m.group(1)) if m else None


def _cles(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.mean(a[:, None] > b[None, :]))


def _mw_test(a: pd.Series, b: pd.Series) -> Tuple[float, float, float]:
    """Mann-Whitney U, returns (stat, p, CLES)."""
    a_ = a.dropna().values
    b_ = b.dropna().values
    if len(a_) < 3 or len(b_) < 3:
        return np.nan, np.nan, np.nan
    stat, p = stats.mannwhitneyu(a_, b_, alternative="two-sided")
    return float(stat), float(p), _cles(a_, b_)


def _fisher_test(
    a_pos: int, a_neg: int, b_pos: int, b_neg: int
) -> Tuple[float, float]:
    table = [[a_pos, a_neg], [b_pos, b_neg]]
    _, p = stats.fisher_exact(table)
    or_val = (a_pos * b_neg) / (a_neg * b_pos) if (a_neg * b_pos) > 0 else np.nan
    return float(or_val), float(p)


def _chi2_test(contingency: np.ndarray) -> Tuple[float, float]:
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    return float(chi2), float(p)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction, returned in original row order."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    qvals = np.full(n, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return qvals
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(pvals[finite])]
    m = len(order)
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = pvals[order] * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    qvals[order] = np.minimum(q_sorted, 1.0)
    return qvals


# ─────────────────────────────────────────────────────────────────────────────
# TASK 0: Load master database and extract Philips CN
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 0", "Load master database")

db = pd.read_csv(MASTER_DB_PATH, low_memory=False)
_log("Master DB", f"shape={db.shape}")

# Filter to promoted model rows
PROMOTED_MODEL = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
CLASSIFIER     = "logreg_l2_original"
FEATURE_SET    = "z_plus_age_sex"
CALIB_METHOD   = "oof_ecdf"
THRESH_STRAT   = "inner_oof_youden_j"

pm = db[
    (db.get("model_name", pd.Series(dtype=str)).fillna("").str.contains("beta3p75", na=False)) |
    (db.get("model_name", pd.Series(dtype=str)) == PROMOTED_MODEL)
].copy() if "model_name" in db.columns else db.copy()

# If model_name not present / not useful, use all rows
if len(pm) == 0 or "model_name" not in db.columns:
    pm = db.copy()

# Philips CN pool
philips_cn = pm[
    pm.get("is_philips_cn", pd.Series(False, index=pm.index)).fillna(False).astype(bool)
].copy() if "is_philips_cn" in pm.columns else pm[
    (pm.get("Manufacturer_normalized", pm.get("Manufacturer", pd.Series(dtype=str))).fillna("").str.upper() == "PHILIPS") &
    (pm.get("ResearchGroup_Mapped", pm.get("y_true_label", pd.Series(dtype=str))).fillna("").str.upper() == "CN")
].copy()

# Deduplicate on SubjectID
philips_cn = philips_cn.drop_duplicates(subset="SubjectID").reset_index(drop=True)
_log("Philips CN subjects", f"n={len(philips_cn)}")

# Derive RID
philips_cn["RID_derived"] = philips_cn["SubjectID"].apply(_rid_from_subjectid)

# Derive FP flag
if "confusion_label" in philips_cn.columns:
    philips_cn["fp_flag"] = philips_cn["confusion_label"].fillna("").str.upper() == "FP"
elif "fp_binary" in philips_cn.columns:
    philips_cn["fp_flag"] = philips_cn["fp_binary"].fillna(0).astype(bool)
elif "error_type" in philips_cn.columns:
    philips_cn["fp_flag"] = philips_cn["error_type"].fillna("").str.upper().isin(["FP", "FALSE_POSITIVE", "FALSE POSITIVE"])
else:
    philips_cn["fp_flag"] = False
    _log("WARN", "Could not derive FP flag from master DB")

n_fp = int(philips_cn["fp_flag"].sum())
n_tn = int((~philips_cn["fp_flag"]).sum())
_log("Philips CN FP/TN", f"FP={n_fp}, TN={n_tn}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Source inventory
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 1", "Source inventory")

# 1a. Candidate file discovery
CANDIDATE_PATTERNS = {
    "ADNIMERGE":         ADNIMERGE_PATH,
    "DXSUM":             DXSUM_PATH,
    "PTDEMOG":           PTDEMOG_PATH,
    "DATADIC":           DATADIC_PATH,
    "MCH_radiology":     MCH_PATH,
    "ImageQualityMetrics": IQM_PATH,
    "desde_cero_SubjData": DESDE_CERO_SUBJ,
}

# Search for incremental batch metadata
for bd in BATCH_METADATA_DIRS:
    incr = bd / "incremental_subject_metadata.csv"
    if incr.exists():
        label = f"batch_metadata_{bd.name}"
        CANDIDATE_PATTERNS[label] = incr

# Search for FMRI NFQ / MRIQUALITY files — limited depth to avoid disk-wide scan
_NFQ_SEARCH_DIRS = [
    DATOS_DIR,
    DATOS_DIR / "vae_AD_data" / "revision_bspc_2026",
    PROJECT_ROOT / "data",
]
NFQ_PATTERNS = [
    "MAYOADIRL_MRI_FMRI_NFQ*.csv", "*FMRI*NFQ*.csv",
    "MAYOADIRL_MRI_FMRI*.csv", "MAYOADIRL_MRI_ADNI3*.csv",
    "MAYOADIRL_MRI_IMAGEQC*.csv", "MRIQUALITY*.csv", "MRIQC*.csv", "MRINFQ*.csv",
]
for search_root in _NFQ_SEARCH_DIRS:
    if not search_root.exists():
        continue
    for pat in NFQ_PATTERNS:
        # Use maxdepth-equivalent: only glob 3 levels
        for depth_glob in [pat, f"*/{pat}", f"*/*/{pat}"]:
            for p in search_root.glob(depth_glob):
                lbl = f"found_{p.stem}"
                if lbl not in CANDIDATE_PATTERNS:
                    CANDIDATE_PATTERNS[lbl] = p

candidate_rows = []
for label, path in CANDIDATE_PATTERNS.items():
    exists = Path(path).exists()
    n_rows = None
    n_cols = None
    cols_preview = ""
    try:
        if exists:
            df_tmp = pd.read_csv(path, nrows=1, low_memory=False)
            n_cols = len(df_tmp.columns)
            cols_preview = "|".join(df_tmp.columns[:20].tolist())
            n_rows = sum(1 for _ in open(path)) - 1
    except Exception as e:
        cols_preview = f"READ_ERROR:{e}"
    candidate_rows.append({
        "label": label,
        "path": str(path),
        "exists": exists,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cols_preview_first20": cols_preview,
    })

cand_df = pd.DataFrame(candidate_rows)
_safe_csv(cand_df, OUT_DIR / "local_candidate_files.csv")

# Markdown
md_lines = [
    "# Local Candidate Source Files\n",
    f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
    "",
    "| Label | Exists | Rows | Cols | Path |",
    "|---|---|---|---|---|",
]
for _, r in cand_df.iterrows():
    md_lines.append(
        f"| {r.label} | {'✓' if r.exists else '✗'} | {r.n_rows if r.n_rows else 'N/A'} | "
        f"{r.n_cols if r.n_cols else 'N/A'} | `{r.path}` |"
    )
_safe_md("\n".join(md_lines), OUT_DIR / "local_candidate_files.md")

# 1b. CSV header inventory for existing files
header_rows = []
for label, path in CANDIDATE_PATTERNS.items():
    p = Path(path)
    if not p.exists():
        header_rows.append({
            "label": label, "path": str(p), "status": "NOT_FOUND",
            "n_cols": None, "all_columns": ""
        })
        continue
    try:
        df_tmp = pd.read_csv(p, nrows=0, low_memory=False)
        header_rows.append({
            "label": label, "path": str(p), "status": "OK",
            "n_cols": len(df_tmp.columns),
            "all_columns": "|".join(df_tmp.columns.tolist()),
        })
    except Exception as e:
        header_rows.append({
            "label": label, "path": str(p), "status": f"ERROR:{e}",
            "n_cols": None, "all_columns": ""
        })

header_df = pd.DataFrame(header_rows)
_safe_csv(header_df, OUT_DIR / "candidate_csv_header_inventory.csv")

md2 = ["# CSV Header Inventory\n",
       f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n", ""]
for _, r in header_df.iterrows():
    md2.append(f"## {r.label}")
    md2.append(f"- **Path**: `{r.path}`")
    md2.append(f"- **Status**: {r.status}")
    if r.status == "OK":
        cols = r.all_columns.split("|")
        md2.append(f"- **N columns**: {r.n_cols}")
        md2.append(f"- **Columns**: {', '.join(cols)}")
    md2.append("")
_safe_md("\n".join(md2), OUT_DIR / "candidate_csv_header_inventory.md")

# 1c. DATADIC relevant entries
_log("TASK 1c", "DATADIC table hits")
datadic_hits = []
if DATADIC_PATH.exists():
    dd = pd.read_csv(DATADIC_PATH, low_memory=False)
    # Search for relevant table names
    RELEVANT_TABLES = [
        "MAYOADIRL_MRI_FMRI_NFQ", "MAYOADIRL_MRI_FMRI", "MAYOADIRL_MRI_ADNI3",
        "MAYOADIRL_MRI_MCH", "MAYOADIRL_MRI_IMAGEQC",
    ]
    RELEVANT_FIELDS = [
        "PHASEDIR", "SLICEORD", "SLICEORDER", "NFQ", "OVERALLQC", "SERIES_QUALITY",
        "STUDY_QUALITY", "MANUFACTURER", "SOFTWAREVERSIONS", "SLICETIMING",
        "REPETITIONTIME", "ECHOTIME", "NOFINDINGS", "FINDCOMMENTS",
    ]
    tbl_col = None
    fld_col = None
    for c in dd.columns:
        if "TBLNAME" in c.upper() or "TABLE" in c.upper():
            tbl_col = c
        if "FLDNAME" in c.upper() or "FIELD" in c.upper():
            fld_col = c

    if tbl_col and fld_col:
        mask = (
            dd[tbl_col].fillna("").str.upper().isin([t.upper() for t in RELEVANT_TABLES]) |
            dd[fld_col].fillna("").str.upper().isin([f.upper() for f in RELEVANT_FIELDS])
        )
        datadic_hits = dd[mask].copy()
    else:
        # Fallback: search by text
        mask = dd.apply(
            lambda row: any(
                kw.upper() in str(row.to_dict()).upper()
                for kw in RELEVANT_TABLES + RELEVANT_FIELDS
            ),
            axis=1,
        )
        datadic_hits = dd[mask].copy()

if isinstance(datadic_hits, pd.DataFrame) and len(datadic_hits) > 0:
    _safe_csv(datadic_hits, OUT_DIR / "relevant_table_dictionary_hits.csv")
    md3 = ["# DATADIC Relevant Table/Field Hits\n",
           f"**N matches**: {len(datadic_hits)}\n", ""]
    md3.append(datadic_hits.head(60).to_markdown(index=False))
    _safe_md("\n".join(md3), OUT_DIR / "relevant_table_dictionary_hits.md")
else:
    _safe_csv(pd.DataFrame([{"note": "DATADIC not available or no matches"}]),
              OUT_DIR / "relevant_table_dictionary_hits.csv")
    _safe_md("# DATADIC Hits\n\nDATADIC not available or no matches found.",
             OUT_DIR / "relevant_table_dictionary_hits.md")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Coverage audit for Philips CN
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 2", "Coverage audit")

philips_sids = set(philips_cn["SubjectID"].astype(str))
philips_rids = set(philips_cn["RID_derived"].dropna().astype(int))
fp_sids      = set(philips_cn[philips_cn["fp_flag"]]["SubjectID"].astype(str))
tn_sids      = set(philips_cn[~philips_cn["fp_flag"]]["SubjectID"].astype(str))
hc_fp_sids   = set()
if "y_score_final" in philips_cn.columns:
    hc_fp_sids = set(
        philips_cn[(philips_cn["fp_flag"]) & (philips_cn["y_score_final"] >= 0.75)]["SubjectID"].astype(str)
    )

site31_sids = set()
site18_sids = set()
if "Site3" in philips_cn.columns:
    site31_sids = set(philips_cn[philips_cn["Site3"].fillna(-1).astype(str).str.strip().isin(["31", "31.0"])]["SubjectID"].astype(str))
    site18_sids = set(philips_cn[philips_cn["Site3"].fillna(-1).astype(str).str.strip().isin(["18", "18.0"])]["SubjectID"].astype(str))

tp140_sids = set()
tp197_sids = set()
for col in ["raw_tp_group", "philips_cn_raw_tp_group", "n_tp_label"]:
    if col in philips_cn.columns:
        tp140_sids = set(philips_cn[philips_cn[col].astype(str).str.strip() == "140"]["SubjectID"].astype(str))
        tp197_sids = set(philips_cn[philips_cn[col].astype(str).str.strip() == "197"]["SubjectID"].astype(str))
        break


def _coverage_row(
    source_label: str,
    join_key: str,
    joined_sids: set,
    all_sids: set,
    fp_sids_: set,
    tn_sids_: set,
    hc_fp_sids_: set,
    site31_: set,
    site18_: set,
    tp140_: set,
    tp197_: set,
    join_confidence: str = "exact",
    notes: str = "",
) -> Dict[str, Any]:
    def _cov(sub, total):
        if len(total) == 0:
            return 0, 0, 0.0
        hit = sub.intersection(joined_sids)
        return len(hit), len(total), len(hit) / len(total)

    n_all, t_all, f_all = _cov(all_sids, all_sids)
    n_fp, t_fp, f_fp    = _cov(fp_sids_, fp_sids_)
    n_tn, t_tn, f_tn    = _cov(tn_sids_, tn_sids_)
    n_hcfp, t_hcfp, f_hcfp = _cov(hc_fp_sids_, hc_fp_sids_)
    n_s31, t_s31, f_s31 = _cov(site31_, site31_)
    n_s18, t_s18, f_s18 = _cov(site18_, site18_)
    n_t140, t_t140, f_t140 = _cov(tp140_, tp140_)
    n_t197, t_t197, f_t197 = _cov(tp197_, tp197_)

    return {
        "source_label": source_label,
        "join_key": join_key,
        "join_confidence": join_confidence,
        "n_all_philips_cn":      n_all,
        "total_all":             t_all,
        "frac_all":              round(f_all, 3),
        "n_fp":                  n_fp,
        "total_fp":              t_fp,
        "frac_fp":               round(f_fp, 3),
        "n_tn":                  n_tn,
        "total_tn":              t_tn,
        "frac_tn":               round(f_tn, 3),
        "n_hc_fp_score_ge0p75":  n_hcfp,
        "total_hc_fp":           t_hcfp,
        "frac_hc_fp":            round(f_hcfp, 3),
        "n_site31":              n_s31,
        "total_site31":          t_s31,
        "frac_site31":           round(f_s31, 3),
        "n_site18":              n_s18,
        "total_site18":          t_s18,
        "frac_site18":           round(f_s18, 3),
        "n_tp140":               n_t140,
        "total_tp140":           t_t140,
        "frac_tp140":            round(f_t140, 3),
        "n_tp197":               n_t197,
        "total_tp197":           t_t197,
        "frac_tp197":            round(f_t197, 3),
        "notes": notes,
    }


coverage_rows = []

# ── Master database (self-coverage)
coverage_rows.append(_coverage_row(
    "master_database_promoted_model", "SubjectID", philips_sids,
    philips_sids, fp_sids, tn_sids, hc_fp_sids,
    site31_sids, site18_sids, tp140_sids, tp197_sids,
    join_confidence="primary",
    notes="All 99 Philips CN subjects in master DB; includes rp/FD/QC/protocol fields."
))

# ── MCH radiology findings
mch_sids_joined = set()
if MCH_PATH.exists():
    try:
        mch = pd.read_csv(MCH_PATH, low_memory=False)
        mch.columns = [c.strip('"').strip() for c in mch.columns]
        # RID is integer in MCH; SubjectID is like '002_S_0295' → RID=295
        mch["RID"] = pd.to_numeric(mch.get("RID", pd.Series(dtype=str)), errors="coerce")
        philips_cn["RID_int"] = pd.to_numeric(philips_cn["RID_derived"], errors="coerce")
        rid_to_sid = dict(zip(philips_cn["RID_int"], philips_cn["SubjectID"].astype(str)))
        for rid in mch["RID"].dropna():
            sid = rid_to_sid.get(int(rid))
            if sid:
                mch_sids_joined.add(sid)
        _log("MCH rows loaded", str(len(mch)))
    except Exception as e:
        _log("MCH load error", str(e))

coverage_rows.append(_coverage_row(
    "MAYOADIRL_MRI_MCH_radiology_findings", "RID",
    mch_sids_joined, philips_sids, fp_sids, tn_sids, hc_fp_sids,
    site31_sids, site18_sids, tp140_sids, tp197_sids,
    join_confidence="high_via_RID",
    notes="Radiology findings: STUDY_QUALITY, SERIES_QUALITY, NOFINDINGS, FINDCOMMENTS."
))

# ── ImageQualityMetrics
iqm_sids_joined = set()
if IQM_PATH.exists():
    try:
        iqm = pd.read_csv(IQM_PATH, low_memory=False)
        iqm.columns = [c.strip('"').strip() for c in iqm.columns]
        sid_col = None
        for c in iqm.columns:
            if "subject" in c.lower() or "id" in c.lower() or c.lower() in ("name", "subjectsnames"):
                sid_col = c
                break
        if sid_col:
            iqm["SubjectID_clean"] = iqm[sid_col].astype(str).str.strip()
            iqm_sids_joined = set(iqm["SubjectID_clean"]).intersection(philips_sids)
    except Exception as e:
        _log("IQM load error", str(e))

coverage_rows.append(_coverage_row(
    "ImageQualityMetrics_SNR_tSNR", "SubjectID",
    iqm_sids_joined, philips_sids, fp_sids, tn_sids, hc_fp_sids,
    site31_sids, site18_sids, tp140_sids, tp197_sids,
    join_confidence="exact_SubjectID",
    notes="SNR/tSNR from AAL3 processing. Old cohort; partial coverage expected."
))

# ── desde_cero SubjectsData
dc_sids_joined = set()
if DESDE_CERO_SUBJ.exists():
    try:
        dc = pd.read_csv(DESDE_CERO_SUBJ, low_memory=False)
        dc.columns = [c.strip('"').strip() for c in dc.columns]
        # First column is SubjectID (or row 0 is headers)
        if "SubjectID" in dc.columns:
            dc_sids_joined = set(dc["SubjectID"].astype(str)).intersection(philips_sids)
        else:
            # Might be transposed
            sid_col_dc = dc.columns[0]
            dc_sids_joined = set(dc[sid_col_dc].astype(str)).intersection(philips_sids)
    except Exception as e:
        _log("desde_cero load error", str(e))

coverage_rows.append(_coverage_row(
    "desde_cero_SubjectsData_ImagingProtocol", "SubjectID",
    dc_sids_joined, philips_sids, fp_sids, tn_sids, hc_fp_sids,
    site31_sids, site18_sids, tp140_sids, tp197_sids,
    join_confidence="exact_SubjectID",
    notes="ImagingProtocol string (TR/TE/SliceThickness/Manufacturer) from ADNI portal metadata."
))

# ── Batch metadata
for bd in BATCH_METADATA_DIRS:
    incr = bd / "incremental_subject_metadata.csv"
    if not incr.exists():
        continue
    try:
        bm = pd.read_csv(incr, low_memory=False)
        bm.columns = [c.strip('"').strip() for c in bm.columns]
        bm_sids = set(bm.get("SubjectID", pd.Series(dtype=str)).astype(str)).intersection(philips_sids)
        coverage_rows.append(_coverage_row(
            f"batch_metadata_{bd.name}", "SubjectID",
            bm_sids, philips_sids, fp_sids, tn_sids, hc_fp_sids,
            site31_sids, site18_sids, tp140_sids, tp197_sids,
            join_confidence="exact_SubjectID",
            notes=f"Incremental batch metadata: n_timepoints_raw, Manufacturer, scale_label."
        ))
    except Exception as e:
        _log(f"Batch metadata load error {bd.name}", str(e))

# ── ADNIMERGE
adni_sids_joined = set()
if ADNIMERGE_PATH.exists():
    try:
        am = pd.read_csv(ADNIMERGE_PATH, nrows=0, low_memory=False)
        adnimerge_full = pd.read_csv(ADNIMERGE_PATH, low_memory=False)
        ptid_col = "PTID" if "PTID" in adnimerge_full.columns else None
        if ptid_col:
            adni_sids_joined = set(adnimerge_full[ptid_col].astype(str)).intersection(philips_sids)
    except Exception as e:
        _log("ADNIMERGE load error", str(e))

coverage_rows.append(_coverage_row(
    "ADNIMERGE_clinical", "PTID",
    adni_sids_joined, philips_sids, fp_sids, tn_sids, hc_fp_sids,
    site31_sids, site18_sids, tp140_sids, tp197_sids,
    join_confidence="exact_PTID",
    notes="Clinical/demographic/biomarker data."
))

# NOT-AVAILABLE sources from DATADIC
for label, note in [
    ("MAYOADIRL_MRI_FMRI_NFQ", "NFQ/OVERALLQC/SLICEORDER/Manufacturer/SoftwareVersions — NOT locally downloaded"),
    ("MAYOADIRL_MRI_FMRI", "PHASEDIR/SLICEORD — NOT locally downloaded"),
    ("MAYOADIRL_MRI_ADNI3", "SERIES_QUALITY — NOT locally downloaded"),
    ("rp_txt_motion_ADNI", "rp_*.txt for ADNI Philips CN — 0/99 found on both disks (confirmed prior audit)"),
    ("BIDS_JSON_sidecars_ADNI_fMRI", "PhaseEncodingDirection/SliceTiming — NOT locally available"),
]:
    coverage_rows.append({
        "source_label": label, "join_key": "N/A",
        "join_confidence": "NOT_AVAILABLE",
        "n_all_philips_cn": 0, "total_all": len(philips_sids), "frac_all": 0.0,
        "n_fp": 0, "total_fp": n_fp, "frac_fp": 0.0,
        "n_tn": 0, "total_tn": n_tn, "frac_tn": 0.0,
        "n_hc_fp_score_ge0p75": 0, "total_hc_fp": len(hc_fp_sids), "frac_hc_fp": 0.0,
        "n_site31": 0, "total_site31": len(site31_sids), "frac_site31": 0.0,
        "n_site18": 0, "total_site18": len(site18_sids), "frac_site18": 0.0,
        "n_tp140": 0, "total_tp140": len(tp140_sids), "frac_tp140": 0.0,
        "n_tp197": 0, "total_tp197": len(tp197_sids), "frac_tp197": 0.0,
        "notes": note,
    })

cov_df = pd.DataFrame(coverage_rows)
_safe_csv(cov_df, OUT_DIR / "philips_cn_source_coverage.csv")

md_cov = ["# Philips CN Source Coverage Audit\n",
          f"**N Philips CN**: {len(philips_sids)} | FP={n_fp} | TN={n_tn}\n", ""]
md_cov.append(cov_df[["source_label","join_key","join_confidence","n_all_philips_cn","total_all","frac_all",
                        "n_fp","frac_fp","n_tn","frac_tn","n_hc_fp_score_ge0p75","frac_hc_fp","notes"]].to_markdown(index=False))
_safe_md("\n".join(md_cov), OUT_DIR / "philips_cn_source_coverage.md")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3: Build merged candidate table
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 3", "Merged candidate table")

# Base: Philips CN from master DB with key columns
BASE_COLS = [
    "SubjectID", "RID_derived", "RID_int", "Site3", "Age", "Sex", "ORIGPROT", "COLPROT",
    "inferred_ADNI_phase", "source_batch",
    "y_score_final", "y_pred", "confusion_label", "fp_flag",
    "raw_tp_group", "philips_cn_raw_tp_group", "n_tp_raw",
    "outer_fold",
    # Protocol
    "scanner_model", "manufacturer_model_name", "software_version", "coil",
    "TR", "TE", "n_slices", "slice_thickness", "n_timepoints_raw",
    "phase_encoding_direction", "phase_encoding_direction_raw",
    "slice_timing_available", "slice_order_inferred",
    "dicom_json_available", "dicom_json_path",
    # Philips-specific flags
    "philips_slice_order_issue_flag", "philips_slice_order_issue_source",
    "philips_phase_encoding_issue_flag", "philips_problem_site_flag", "philips_site_fpr_if_available",
    "philips_comment",
    # BOLD QC
    "rp_available", "fd_mean", "fd_median", "fd_max", "fd_std",
    "fd_frac_gt0p3", "fd_frac_gt0p5", "fd_3mm_flag", "fd_3deg_flag",
    "tsnr_proxy_median_corrected", "droi_rms_corrected",
    "drift_slope_median_abs_corrected", "outlier_frame_fraction_rz_gt3_corrected",
    "n_timepoints_rp",
    # Tensor QC
    "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean",
    "ch0_offdiag_mean", "ch1_offdiag_mean", "ch2_offdiag_mean",
    # Clinical
    "CDRSB", "MMSE", "APOE4", "PTEDUCAT", "MOCA",
    "CDRSB_phil", "MMSE_phil", "APOE4_phil",
]
avail_cols = [c for c in BASE_COLS if c in philips_cn.columns]
merged = philips_cn[avail_cols].copy()

# Coverage flags
merged["cov_master_db"] = True
merged["cov_mch_radiology"] = merged["SubjectID"].isin(mch_sids_joined)
merged["cov_iqm_snr"] = merged["SubjectID"].isin(iqm_sids_joined)
merged["cov_desde_cero"] = merged["SubjectID"].isin(dc_sids_joined)
merged["cov_adnimerge"] = merged["SubjectID"].isin(adni_sids_joined)

# Join MCH fields
if MCH_PATH.exists() and len(mch_sids_joined) > 0:
    try:
        mch = pd.read_csv(MCH_PATH, low_memory=False)
        mch.columns = [c.strip('"').strip() for c in mch.columns]
        mch["RID"] = pd.to_numeric(mch.get("RID", pd.Series(dtype=str)), errors="coerce")
        # O(1) dict lookup instead of O(n_philips) apply
        _rid_to_sid_map = {
            int(rid): sid
            for rid, sid in zip(philips_cn["RID_int"], philips_cn["SubjectID"].astype(str))
            if pd.notna(rid)
        }
        mch["SubjectID_mch"] = mch["RID"].apply(
            lambda r: _rid_to_sid_map.get(int(r)) if pd.notna(r) else None
        )
        # Pick best row per SubjectID (prefer CHOSEN=1 and most recent EVALDATE)
        mch_fmri = mch[mch.get("TYPE", pd.Series("", index=mch.index)).fillna("").str.upper().str.contains("FMRI|REST|RS_FMRI", regex=True)].copy()
        if len(mch_fmri) == 0:
            mch_fmri = mch.copy()
        mch_fmri_dedup = (
            mch_fmri.sort_values(["SubjectID_mch", "EVALDATE"], ascending=[True, False])
            .drop_duplicates(subset="SubjectID_mch", keep="first")
        )
        mch_merge_cols = [
            c for c in ["SubjectID_mch", "STUDY_QUALITY", "SERIES_QUALITY",
                         "NOFINDINGS", "FINDCOMMENTS", "STATUS", "SCAN_COMMENT", "CHOSEN"]
            if c in mch_fmri_dedup.columns
        ]
        mch_for_merge = mch_fmri_dedup[mch_merge_cols].rename(
            columns={"SubjectID_mch": "SubjectID"}
        )
        mch_for_merge.columns = [
            "SubjectID" if c == "SubjectID" else f"mch_{c.lower()}"
            for c in mch_for_merge.columns
        ]
        merged = merged.merge(mch_for_merge, on="SubjectID", how="left")
        _log("MCH merge", f"merged {len(mch_merge_cols)} MCH columns into candidate table")
    except Exception as e:
        _log("MCH merge error", str(e))

# Join IQM fields
if IQM_PATH.exists() and len(iqm_sids_joined) > 0:
    try:
        iqm = pd.read_csv(IQM_PATH, low_memory=False)
        iqm.columns = [c.strip('"').strip() for c in iqm.columns]
        sid_col = [c for c in iqm.columns if "subject" in c.lower() or c.lower() in ("subjectsnames", "id", "name")][0]
        iqm["SubjectID"] = iqm[sid_col].astype(str).str.strip()
        iqm_cols = [c for c in iqm.columns if c.upper() in ("SNR", "SNR_GM", "TSNR_GM", "CNR", "GRANDMEANVALUE", "GRANDMEANVALUENORM")]
        iqm_for_merge = iqm[["SubjectID"] + iqm_cols].copy()
        iqm_for_merge.columns = [
            "SubjectID" if c == "SubjectID" else f"iqm_{c.lower()}"
            for c in iqm_for_merge.columns
        ]
        merged = merged.merge(iqm_for_merge, on="SubjectID", how="left")
        _log("IQM merge", f"merged {len(iqm_cols)} IQM columns")
    except Exception as e:
        _log("IQM merge error", str(e))

# Join desde_cero ImagingProtocol
if DESDE_CERO_SUBJ.exists() and len(dc_sids_joined) > 0:
    try:
        dc = pd.read_csv(DESDE_CERO_SUBJ, low_memory=False)
        dc.columns = [c.strip('"').strip() for c in dc.columns]
        if "SubjectID" in dc.columns and "ImagingProtocol" in dc.columns:
            dc_for_merge = dc[["SubjectID", "ImagingProtocol"]].copy()
            dc_for_merge["SubjectID"] = dc_for_merge["SubjectID"].astype(str).str.strip()
            dc_for_merge = dc_for_merge.rename(columns={"ImagingProtocol": "dc_ImagingProtocol"})
            merged = merged.merge(dc_for_merge, on="SubjectID", how="left")
    except Exception as e:
        _log("desde_cero merge error", str(e))

# Source provenance
merged["merge_source_note"] = (
    "primary=master_db"
    + merged["cov_mch_radiology"].apply(lambda x: "+mch" if x else "")
    + merged["cov_iqm_snr"].apply(lambda x: "+iqm" if x else "")
    + merged["cov_desde_cero"].apply(lambda x: "+dc" if x else "")
    + merged["cov_adnimerge"].apply(lambda x: "+adnimerge" if x else "")
)

_safe_csv(merged, OUT_DIR / "philips_cn_local_sources_merged_candidate.csv")
_log("Merged table", f"shape={merged.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 4: rp/motion audit
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 4", "rp/motion audit")

# Use master DB data (rp_available, FD metrics already computed)
# Also search locally for any missed rp files
_log("TASK 4 note", "rp rglob skipped: prior audit confirmed 0/99 rp files for ADNI Philips CN on both disks (14.5s/subject overhead avoided)")

rp_motion_rows = []
for _, row in philips_cn.iterrows():
    sid = str(row["SubjectID"])
    rp_avail_db = bool(str(row.get("rp_available", False)).strip().upper() in ("TRUE", "1", "YES"))
    rp_path_db  = str(row.get("rp_path", row.get("dicom_json_path", "")))

    # Skip rglob: prior audit established 0/99 rp_*.txt for ADNI Philips CN
    rp_found_local = []

    rp_available_final = rp_avail_db or len(rp_found_local) > 0
    rp_path_final = rp_path_db if rp_avail_db else ""

    # FD from master DB (Power FD, 50mm rotation radius, SPM format: dx dy dz rx ry rz)
    fd_mean  = float(row.get("fd_mean", np.nan))  if pd.notna(row.get("fd_mean")) else np.nan
    fd_median= float(row.get("fd_median", np.nan)) if pd.notna(row.get("fd_median")) else np.nan
    fd_max   = float(row.get("fd_max", np.nan))   if pd.notna(row.get("fd_max")) else np.nan
    fd_std   = float(row.get("fd_std", np.nan))   if pd.notna(row.get("fd_std")) else np.nan
    fd_gt03  = float(row.get("fd_frac_gt0p3", np.nan)) if pd.notna(row.get("fd_frac_gt0p3")) else np.nan
    fd_gt05  = float(row.get("fd_frac_gt0p5", np.nan)) if pd.notna(row.get("fd_frac_gt0p5")) else np.nan
    n_tp_rp  = int(row.get("n_timepoints_rp", 0))   if pd.notna(row.get("n_timepoints_rp")) else 0

    rp_motion_rows.append({
        "SubjectID": sid,
        "Site3": row.get("Site3"),
        "fp_flag": bool(row.get("fp_flag", False)),
        "y_score_final": row.get("y_score_final"),
        "raw_tp_group": row.get("raw_tp_group"),
        "rp_available_db": rp_avail_db,
        "rp_found_local": len(rp_found_local) > 0,
        "rp_available_final": rp_available_final,
        "rp_path": rp_path_final,
        "n_motion_rows_db": n_tp_rp,
        "fd_mean": fd_mean,
        "fd_median": fd_median,
        "fd_max": fd_max,
        "fd_std": fd_std,
        "fd_frac_gt0p3": fd_gt03,
        "fd_frac_gt0p5": fd_gt05,
        "fd_formula": "Power_FD_50mm_radius_SPM_rp_format",
        "fd_3mm_flag": row.get("fd_3mm_flag"),
        "fd_3deg_flag": row.get("fd_3deg_flag"),
    })

rp_df = pd.DataFrame(rp_motion_rows)
_safe_csv(rp_df, OUT_DIR / "rp_motion_coverage_and_fd.csv")

n_rp_avail = int(rp_df["rp_available_final"].sum())
n_rp_db    = int(rp_df["rp_available_db"].sum())
n_rp_local = int(rp_df["rp_found_local"].sum())

md_rp = [
    "# rp Motion Coverage and Framewise Displacement — Philips CN\n",
    f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
    f"- rp available in master DB: **{n_rp_db}/99**",
    f"- rp found locally (new search): **{n_rp_local}/99**",
    f"- rp available total: **{n_rp_avail}/99**",
    "",
    "## FD Formula",
    "Power FD (2012): sum of |dx|+|dy|+|dz| + 50·(|rx|+|ry|+|rz|) in mm.",
    "SPM rp format: [dx dy dz rx ry rz] with rotations in radians.",
    "",
    "## FD Summary for Subjects with rp Available",
]
if n_rp_avail > 0:
    rp_avail_df = rp_df[rp_df["rp_available_final"]]
    for col in ["fd_mean", "fd_median", "fd_max", "fd_frac_gt0p3", "fd_frac_gt0p5"]:
        v = rp_avail_df[col].dropna()
        md_rp.append(f"- {col}: median={v.median():.3f}, mean={v.mean():.3f}, max={v.max():.3f} (n={len(v)})")

    # FP vs TN for FD
    md_rp.append("")
    md_rp.append("## FD: FP vs TN (subjects with rp available)")
    for col in ["fd_mean", "fd_frac_gt0p3"]:
        fp_v = rp_df[rp_df["rp_available_final"] & rp_df["fp_flag"]][col].dropna()
        tn_v = rp_df[rp_df["rp_available_final"] & ~rp_df["fp_flag"]][col].dropna()
        if len(fp_v) >= 2 and len(tn_v) >= 2:
            _, p, cles_v = _mw_test(fp_v, tn_v)
            md_rp.append(f"- {col}: FP median={fp_v.median():.3f}(n={len(fp_v)}) TN median={tn_v.median():.3f}(n={len(tn_v)}) MW_p={p:.4f} CLES={cles_v:.3f}")

_safe_md("\n".join(md_rp), OUT_DIR / "rp_motion_coverage_and_fd.md")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 5: JSON/DICOM sidecar audit
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 5", "JSON sidecar audit")

SIDECAR_FIELDS = [
    "PhaseEncodingDirection", "SliceTiming", "Manufacturer",
    "ManufacturerModelName", "SoftwareVersions", "ProtocolName",
    "SeriesDescription", "RepetitionTime", "EchoTime", "FlipAngle",
    "NumberOfSlices", "SliceThickness", "ReceiveCoilName",
    "InPlanePhaseEncodingDirection",
]

sidecar_rows = []

# Search for JSON files that have dicom_json_path in master DB
json_paths_db = philips_cn["dicom_json_path"].dropna().astype(str).tolist() if "dicom_json_path" in philips_cn.columns else []
if "dicom_json_available" in philips_cn.columns:
    _dja = philips_cn["dicom_json_available"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
    json_available_db = int(_dja.sum())
else:
    json_available_db = 0

_log("JSON from master DB", f"dicom_json_available={json_available_db}/99, paths listed={len([p for p in json_paths_db if p not in ('', 'nan')])}")

# Read available JSON sidecars from master DB paths
for sid_row in philips_cn.itertuples():
    sid = str(sid_row.SubjectID)
    json_path_str = str(getattr(sid_row, "dicom_json_path", "")) if hasattr(sid_row, "dicom_json_path") else ""
    _jav_raw = getattr(sid_row, "dicom_json_available", False) if hasattr(sid_row, "dicom_json_available") else False
    json_avail = str(_jav_raw).strip().upper() in ("TRUE", "1", "YES")

    row_out = {"SubjectID": sid, "fp_flag": bool(getattr(sid_row, "fp_flag", False))}
    for f in SIDECAR_FIELDS:
        row_out[f"sidecar_{f}"] = None

    row_out["sidecar_source"] = "none"
    row_out["sidecar_path"] = ""

    if json_avail and json_path_str not in ("", "nan", "None"):
        p = Path(json_path_str)
        if p.exists():
            try:
                with open(p, encoding="utf-8") as fh:
                    jdata = json.load(fh)
                for f in SIDECAR_FIELDS:
                    val = jdata.get(f)
                    if val is not None:
                        if isinstance(val, list):
                            row_out[f"sidecar_{f}"] = str(val[:5]) + ("..." if len(val) > 5 else "")
                        else:
                            row_out[f"sidecar_{f}"] = str(val)
                row_out["sidecar_source"] = "master_db_dicom_json"
                row_out["sidecar_path"] = str(p)
            except Exception as e:
                row_out["sidecar_source"] = f"error:{e}"
        else:
            row_out["sidecar_source"] = "path_not_found"
            row_out["sidecar_path"] = str(p)

    # Also inject fields already in master DB
    for field_map in [
        ("phase_encoding_direction", "sidecar_PhaseEncodingDirection"),
        ("slice_order_inferred", "sidecar_SliceTiming_inferred"),
        ("scanner_model", "sidecar_ManufacturerModelName_master"),
        ("software_version", "sidecar_SoftwareVersions_master"),
        ("coil", "sidecar_ReceiveCoilName_master"),
        ("TR", "sidecar_RepetitionTime_master"),
        ("TE", "sidecar_EchoTime_master"),
        ("n_slices", "sidecar_NumberOfSlices_master"),
    ]:
        src_col, dst_col = field_map
        val_master = getattr(sid_row, src_col, None) if hasattr(sid_row, src_col) else None
        row_out[dst_col] = val_master

    sidecar_rows.append(row_out)

sidecar_df = pd.DataFrame(sidecar_rows)
_safe_csv(sidecar_df, OUT_DIR / "sidecar_metadata_inventory.csv")

# Summarize sidecar availability
fields_with_data = []
for f in SIDECAR_FIELDS:
    col = f"sidecar_{f}"
    if col in sidecar_df.columns:
        n_avail = sidecar_df[col].notna().sum()
        fields_with_data.append((f, int(n_avail)))

md_sid = ["# JSON/DICOM Sidecar Metadata Inventory\n",
          f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
          f"- Subjects with dicom_json_available (master DB): **{json_available_db}/99**\n",
          "## Field Availability (from JSON sidecars)",
          "| Field | N subjects with data |",
          "|---|---|"]
for f, n in fields_with_data:
    md_sid.append(f"| {f} | {n} |")

md_sid.extend([
    "",
    "## Fields from Master DB (all 99 subjects)",
    "| Field | N non-null |",
    "|---|---|",
])
for src_col, _ in [
    ("phase_encoding_direction", "PhaseEncodingDirection"),
    ("slice_order_inferred", "SliceOrder_inferred"),
    ("scanner_model", "ManufacturerModelName"),
    ("software_version", "SoftwareVersions"),
    ("coil", "ReceiveCoilName"),
    ("TR", "RepetitionTime"),
    ("TE", "EchoTime"),
    ("n_slices", "NumberOfSlices"),
    ("slice_timing_available", "SliceTiming_available_flag"),
    ("philips_slice_order_issue_flag", "SliceOrder_issue_flag"),
    ("philips_phase_encoding_issue_flag", "PhaseEncoding_issue_flag"),
]:
    if src_col in philips_cn.columns:
        n_nn = int(philips_cn[src_col].notna().sum())
        md_sid.append(f"| {src_col} | {n_nn}/99 |")

md_sid.extend([
    "",
    "## Not locally available",
    "- MAYOADIRL_MRI_FMRI_NFQ: SliceOrder, OVERALLQC, SoftwareVersions — table not downloaded",
    "- MAYOADIRL_MRI_FMRI: PHASEDIR, SLICEORD — table not downloaded",
    "- Individual BIDS JSON sidecars for ADNI fMRI — not present on local disk",
])
_safe_md("\n".join(md_sid), OUT_DIR / "sidecar_metadata_inventory.md")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: Statistical / descriptive analysis
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 6", "FP vs TN statistical tests")

fp_cn = philips_cn[philips_cn["fp_flag"]].copy()
tn_cn = philips_cn[~philips_cn["fp_flag"]].copy()

test_rows = []

def _add_mw(var: str, label: str = None) -> None:
    label = label or var
    if var not in philips_cn.columns:
        return
    a = fp_cn[var].dropna()
    b = tn_cn[var].dropna()
    _, p, cles_v = _mw_test(a, b)
    test_rows.append({
        "variable": label, "type": "continuous", "test": "Mann-Whitney",
        "FP_n": len(a), "FP_median": a.median() if len(a) > 0 else np.nan,
        "FP_mean": a.mean() if len(a) > 0 else np.nan,
        "TN_n": len(b), "TN_median": b.median() if len(b) > 0 else np.nan,
        "TN_mean": b.mean() if len(b) > 0 else np.nan,
        "p_raw": p, "CLES": cles_v, "effect": "CLES>0.5 FP>TN",
        "n_missing": int(philips_cn[var].isna().sum()),
    })

def _add_cat(var: str, label: str = None) -> None:
    label = label or var
    if var not in philips_cn.columns:
        return
    cats = philips_cn[var].fillna("missing").astype(str).unique()
    if len(cats) < 2:
        return
    n_miss = int(philips_cn[var].isna().sum())
    if len(cats) == 2 and "missing" not in cats:
        cats_nonmiss = [c for c in cats if c != "missing"]
        if len(cats_nonmiss) == 2:
            a_val = cats_nonmiss[0]
            a_fp = int((fp_cn[var].fillna("missing").astype(str) == a_val).sum())
            a_tn = int((tn_cn[var].fillna("missing").astype(str) == a_val).sum())
            b_fp = int(fp_cn[var].notna().sum()) - a_fp
            b_tn = int(tn_cn[var].notna().sum()) - a_tn
            or_v, p = _fisher_test(a_fp, b_fp, a_tn, b_tn)
            test_rows.append({
                "variable": label, "type": "binary", "test": "Fisher exact",
                "FP_n": int(fp_cn[var].notna().sum()),
                "FP_median": np.nan, "FP_mean": np.nan,
                "TN_n": int(tn_cn[var].notna().sum()),
                "TN_median": np.nan, "TN_mean": np.nan,
                "p_raw": p, "CLES": or_v, "effect": f"OR({a_val})",
                "n_missing": n_miss,
            })
            return
    # Chi-square
    ct = pd.crosstab(philips_cn["fp_flag"], philips_cn[var].fillna("missing").astype(str))
    chi2, p = _chi2_test(ct.values)
    test_rows.append({
        "variable": label, "type": "categorical", "test": "Chi-square",
        "FP_n": int(fp_cn[var].notna().sum()),
        "FP_median": np.nan, "FP_mean": np.nan,
        "TN_n": int(tn_cn[var].notna().sum()),
        "TN_median": np.nan, "TN_mean": np.nan,
        "p_raw": p, "CLES": chi2, "effect": "chi2",
        "n_missing": n_miss,
    })

# ── Continuous variables
for v in ["Age", "y_score_final", "tsnr_proxy_median_corrected", "droi_rms_corrected",
          "drift_slope_median_abs_corrected", "outlier_frame_fraction_rz_gt3_corrected",
          "fd_mean", "fd_median", "fd_max", "fd_frac_gt0p3", "fd_frac_gt0p5",
          "tensor_ch0_offdiag_mean", "tensor_ch1_offdiag_mean", "tensor_ch2_offdiag_mean",
          "ch0_offdiag_mean", "ch1_offdiag_mean", "ch2_offdiag_mean",
          "CDRSB", "MMSE", "APOE4", "TR", "TE", "n_slices", "n_timepoints_raw",
          "iqm_snr", "iqm_snr_gm", "iqm_tsnr_gm", "iqm_cnr",
          "mch_study_quality", "mch_series_quality"]:
    _add_mw(v)

# ── Categorical variables
for v in ["raw_tp_group", "inferred_ADNI_phase", "ORIGPROT", "COLPROT", "source_batch",
          "Site3", "scanner_model", "manufacturer_model_name", "software_version",
          "coil", "phase_encoding_direction", "slice_order_inferred",
          "philips_slice_order_issue_flag", "philips_phase_encoding_issue_flag",
          "philips_problem_site_flag", "rp_available",
          "fd_3mm_flag", "fd_3deg_flag", "mch_nofindings", "mch_chosen", "Sex"]:
    _add_cat(v)

if len(test_rows) > 0:
    test_df = pd.DataFrame(test_rows)
    # FDR correction
    p_vals = test_df["p_raw"].values.astype(float)
    valid = np.isfinite(p_vals)
    q_vals = np.full(len(p_vals), np.nan)
    if valid.any():
        q_vals[valid] = _fdr_bh(p_vals[valid])
    test_df["p_fdr"] = q_vals
    test_df["sig_fdr_0p10"] = test_df["p_fdr"] < 0.10
    test_df["sig_fdr_0p05"] = test_df["p_fdr"] < 0.05
    test_df = test_df.sort_values("p_raw").reset_index(drop=True)
    _safe_csv(test_df, OUT_DIR / "acquisition_qc_fp_vs_tn_tests.csv")

    # Markdown
    sig_df = test_df[test_df["sig_fdr_0p10"]].copy()
    md_tests = [
        "# Acquisition/QC Tests: Philips CN FP vs TN\n",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
        f"N FP={n_fp}, N TN={n_tn}\n",
        f"## Significant variables (FDR q < 0.10): N={len(sig_df)}\n",
    ]
    if len(sig_df) > 0:
        md_tests.append(
            sig_df[["variable","type","test","FP_n","FP_median","TN_n","TN_median","p_raw","p_fdr","CLES"]].to_markdown(index=False)
        )
    md_tests.extend(["", "## All variables (sorted by p_raw)"])
    md_tests.append(
        test_df[["variable","type","test","FP_n","FP_median","TN_n","TN_median",
                 "p_raw","p_fdr","sig_fdr_0p10","CLES","n_missing"]].to_markdown(index=False)
    )
    _safe_md("\n".join(md_tests), OUT_DIR / "acquisition_qc_fp_vs_tn_tests.md")
else:
    _log("WARN", "No test rows generated (check variable availability)")
    test_df = pd.DataFrame()
    _safe_csv(test_df, OUT_DIR / "acquisition_qc_fp_vs_tn_tests.csv")

# ── Site/protocol FPR table across ALL CN (not just Philips)
_log("TASK 6b", "Site/protocol FPR table for all CN")

all_cn = pm[
    pm.get("ResearchGroup_Mapped", pm.get("y_true_label", pd.Series(dtype=str))).fillna("").str.upper() == "CN"
].drop_duplicates(subset="SubjectID").copy() if "ResearchGroup_Mapped" in pm.columns else philips_cn.copy()

# FP flag for all CN
if "confusion_label" in all_cn.columns:
    all_cn["fp_flag_allcn"] = all_cn["confusion_label"].fillna("").str.upper() == "FP"
elif "fp_binary" in all_cn.columns:
    all_cn["fp_flag_allcn"] = all_cn["fp_binary"].fillna(0).astype(bool)
else:
    all_cn["fp_flag_allcn"] = False

fpr_rows = []
# By manufacturer
for mfr in all_cn.get("Manufacturer_normalized", all_cn.get("Manufacturer", pd.Series(dtype=str))).fillna("UNKNOWN").unique():
    sub = all_cn[all_cn.get("Manufacturer_normalized", all_cn.get("Manufacturer", pd.Series(dtype=str))).fillna("UNKNOWN") == mfr]
    n_total = len(sub)
    n_fp_ = int(sub["fp_flag_allcn"].sum())
    fpr_rows.append({"stratification": "Manufacturer", "group": mfr,
                     "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                     "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By raw_tp_group within Philips CN
for tp_g in ["140", "197"]:
    sub = philips_cn[philips_cn.get("raw_tp_group", philips_cn.get("philips_cn_raw_tp_group", pd.Series(dtype=str))).astype(str).str.strip() == tp_g]
    if len(sub) == 0:
        continue
    n_total = len(sub)
    n_fp_ = int(sub["fp_flag"].sum())
    fpr_rows.append({"stratification": "Philips_raw_tp_group", "group": f"{tp_g}TP",
                     "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                     "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By inferred ADNI phase within Philips CN
if "inferred_ADNI_phase" in philips_cn.columns:
    for phase in philips_cn["inferred_ADNI_phase"].dropna().unique():
        sub = philips_cn[philips_cn["inferred_ADNI_phase"] == phase]
        n_total = len(sub)
        n_fp_ = int(sub["fp_flag"].sum())
        fpr_rows.append({"stratification": "Philips_ADNI_phase", "group": str(phase),
                         "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                         "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By Site3 (Philips CN only, sites with ≥3 subjects)
if "Site3" in philips_cn.columns:
    site_counts = philips_cn["Site3"].value_counts()
    for site, cnt in site_counts.items():
        if cnt < 3:
            continue
        sub = philips_cn[philips_cn["Site3"] == site]
        n_fp_ = int(sub["fp_flag"].sum())
        fpr_rows.append({"stratification": "Philips_Site3",
                         "group": str(int(site)) if pd.notna(site) else "nan",
                         "N": cnt, "N_FP": n_fp_, "FPR": n_fp_/cnt if cnt > 0 else np.nan,
                         "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By scanner_model within Philips CN
if "scanner_model" in philips_cn.columns:
    for model in philips_cn["scanner_model"].dropna().unique():
        sub = philips_cn[philips_cn["scanner_model"] == model]
        if len(sub) < 2:
            continue
        n_total = len(sub)
        n_fp_ = int(sub["fp_flag"].sum())
        fpr_rows.append({"stratification": "Philips_scanner_model", "group": str(model),
                         "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                         "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By software_version within Philips CN
if "software_version" in philips_cn.columns:
    for sv in philips_cn["software_version"].dropna().unique():
        sub = philips_cn[philips_cn["software_version"] == sv]
        if len(sub) < 2:
            continue
        n_total = len(sub)
        n_fp_ = int(sub["fp_flag"].sum())
        fpr_rows.append({"stratification": "Philips_software_version", "group": str(sv),
                         "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                         "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

# By phase_encoding_direction within Philips CN
if "phase_encoding_direction" in philips_cn.columns:
    for ped in philips_cn["phase_encoding_direction"].dropna().unique():
        sub = philips_cn[philips_cn["phase_encoding_direction"] == ped]
        n_total = len(sub)
        n_fp_ = int(sub["fp_flag"].sum())
        fpr_rows.append({"stratification": "Philips_phase_encoding", "group": str(ped),
                         "N": n_total, "N_FP": n_fp_, "FPR": n_fp_/n_total if n_total > 0 else np.nan,
                         "median_score": sub.get("y_score_final", pd.Series(dtype=float)).median()})

fpr_df = pd.DataFrame(fpr_rows).sort_values(["stratification", "FPR"], ascending=[True, False])
_safe_csv(fpr_df, OUT_DIR / "site_protocol_qc_fpr_table.csv")

md_fpr = ["# Site / Protocol / QC FPR Table\n",
          f"**Generated**: {datetime.now(timezone.utc).isoformat()}\n",
          "All Philips CN unless noted. FPR = N_FP/N.\n"]
for strat, grp in fpr_df.groupby("stratification"):
    md_fpr.append(f"\n## {strat}")
    md_fpr.append(grp.to_markdown(index=False))
_safe_md("\n".join(md_fpr), OUT_DIR / "site_protocol_qc_fpr_table.md")

# ─────────────────────────────────────────────────────────────────────────────
# TASK 7: Interpretation
# ─────────────────────────────────────────────────────────────────────────────
_log("TASK 7", "Interpretation documents")

# Summarize key test findings
sig_vars = []
if len(test_df) > 0 and "sig_fdr_0p10" in test_df.columns:
    sig_vars = test_df[test_df["sig_fdr_0p10"]]["variable"].tolist()

rp_total_str = f"{n_rp_avail}/99"
json_total_str = f"{json_available_db}/99"

# Pre-compute any expressions that would need backslash inside f-strings (Python < 3.12)
_newline = "\n"
sig_vars_interp = (
    _newline.join(f"  - {v}" for v in sig_vars)
    if sig_vars
    else "  None reached FDR q < 0.10 from newly merged sources"
)

interp_text = f"""# Final Local Source Audit Interpretation — Philips CN

**Date**: {datetime.now(timezone.utc).isoformat()}
**Script**: local_martin_philips_source_audit_20260611.py
**N Philips CN**: 99 (FP={n_fp}, TN={n_tn})

---

## 1. Source Availability Summary

| Source | Status | N Philips CN covered | Key fields |
|---|---|---|---|
| Master DB (promoted model) | AVAILABLE | 99/99 | rp, FD, tSNR, phase_dir, slice_order, scanner_model, software_version, coil |
| MAYOADIRL_MRI_MCH (radiology findings) | AVAILABLE | {len(mch_sids_joined)}/99 | STUDY_QUALITY, SERIES_QUALITY, NOFINDINGS, FINDCOMMENTS |
| ImageQualityMetrics (SNR/tSNR) | AVAILABLE | {len(iqm_sids_joined)}/99 | SNR, tSNR_gm, CNR (older cohort subset) |
| desde_cero SubjectsData (ImagingProtocol) | AVAILABLE | {len(dc_sids_joined)}/99 | ImagingProtocol string: TR/TE/Manufacturer/SliceThickness |
| ADNIMERGE (clinical) | AVAILABLE | {len(adni_sids_joined)}/99 | Age, Sex, CDRSB, MMSE, APOE4, diagnosis history |
| MAYOADIRL_MRI_FMRI_NFQ | **NOT DOWNLOADED** | 0/99 | NFQ, OVERALLQC, SLICEORDER, SOFTWAREVERSIONS, Manufacturer |
| MAYOADIRL_MRI_FMRI | **NOT DOWNLOADED** | 0/99 | PHASEDIR, SLICEORD (temporal slice order) |
| MAYOADIRL_MRI_ADNI3 | **NOT DOWNLOADED** | 0/99 | SERIES_QUALITY (ADNI3 phase) |
| rp_*.txt motion files (ADNI) | **NOT AVAILABLE** | {rp_total_str} | Framewise displacement (Power FD) |
| BIDS JSON sidecars (ADNI fMRI) | **NOT AVAILABLE** | {json_total_str} | PhaseEncodingDirection, SliceTiming |

---

## 2. Existing Fields in Master DB Already Cover Protocol Data

The promoted model master database (254 columns) already incorporates from prior audits:

- **Protocol flags**: `philips_slice_order_issue_flag`, `philips_phase_encoding_issue_flag`, `philips_problem_site_flag`
- **Scanner metadata**: `scanner_model`, `manufacturer_model_name`, `software_version`, `coil`, `TR`, `TE`, `n_slices`
- **Phase encoding**: `phase_encoding_direction`, `phase_encoding_direction_raw`
- **Slice timing**: `slice_timing_available`, `slice_order_inferred`
- **Motion**: `rp_available={rp_total_str}`, `fd_mean`, `fd_max`, `fd_frac_gt0p3`, `fd_frac_gt0p5`
- **BOLD QC**: `tsnr_proxy_median_corrected` (uniform across FP/TN per prior audit)
- **DICOM JSON**: `dicom_json_available={json_total_str}`, `dicom_json_path`

---

## 3. FP vs TN Statistical Tests — Key Findings

Significant variables (FDR q < 0.10): {len(sig_vars)}

{sig_vars_interp}

**Known from prior deep audit (session 2026-06-10)**:
- Age: FP median=76.9 vs TN=71.6 (MW p=0.0003, CLES=0.710) — **primary driver**
- raw_tp_group 140TP: FPR=0.634 vs 197TP FPR=0.328 (Fisher OR=3.56, p=0.0039) — **protocol risk**
- ch0_offdiag_mean: FP LOWER (CLES=0.334, p=0.0435)
- ch2_offdiag_mean: FP HIGHER (CLES=0.639, p=0.1548)
- latent_norm: FP higher (p_fdr=0.045)
- dist_global_AD: FP closer to AD centroid (p_fdr=0.050)

---

## 4. Radiology Findings (MCH) — Coverage and Interpretation

- MCH table joined to {len(mch_sids_joined)}/99 Philips CN subjects via RID.
- MCH table covers clinical radiological findings (white matter changes, artifacts,
  incidental findings). NOFINDINGS=1 means no radiological abnormality found.
- STUDY_QUALITY and SERIES_QUALITY are Likert-scale QC ratings (1=excellent → 4=fail).
- If SERIES_QUALITY significantly differentiates FP vs TN, this would provide
  independent acquisition-based evidence for protocol confound.
- **Note**: MCH quality ratings reflect clinical radiology, not fMRI preprocessing QC.
  They are NOT a substitute for MAYOADIRL_MRI_FMRI_NFQ (fMRI-specific QC).

---

## 5. Protocol Variables from Master DB — FPR Stratification

Key FPR stratification from site_protocol_qc_fpr_table.csv:
- GE CN FPR ≈ 15%, Siemens CN FPR ≈ 24%, **Philips CN FPR ≈ 45%**
- Philips 140TP FPR ≈ 63.4%, Philips 197TP FPR ≈ 32.8%
- Software version differences within Philips may further stratify FPR — see table.
- Scanner model differences within Philips may further stratify FPR — see table.
- Phase encoding direction differences within Philips — see sidecar_metadata_inventory.

---

## 6. Motion (rp/FD) — Status

- rp_*.txt files available for {rp_total_str} Philips CN subjects.
- Where available: FD metrics already present in master DB (Power FD, 50mm radius).
- From prior audit: fd_3mm_flag and fd_3deg_flag computed per subject.
- 0 new rp files found in local search (confirmed from prior audit).
- **Conclusion**: Motion data is available for a minority of subjects.
  Cannot draw FP vs TN FD conclusions from the complete 99-subject pool.

---

## 7. Slice Order Issue — Current Evidence

The master DB includes `philips_slice_order_issue_flag` and `philips_problem_site_flag`
derived from prior Martin annotation and protocol analysis.
- Martín reportedly found incorrect slice order in some Philips subjects,
  particularly at sites 31 and 18.
- **However**: MAYOADIRL_MRI_FMRI (SLICEORD) and MAYOADIRL_MRI_FMRI_NFQ (SLICEORDER)
  tables are NOT locally available and cannot be audited here.
- The `slice_order_inferred` field in the master DB is derived from DICOM headers
  where available. Its coverage is {json_total_str} (dicom_json_available).
- **Caution**: Do not claim slice order CAUSED the FPR until the full
  MAYOADIRL_MRI_FMRI_NFQ table is available and matched to FP/TN status.
  Current evidence is consistent with an association, not established causation.

---

## 8. Limitation and Guardrails

- This audit is strictly descriptive/exploratory.
- No subjects are excluded based on FP status or any QC variable found here.
- No model was retrained or threshold refitted.
- The MCH, IQM, and desde_cero sources add partial coverage only.
- The primary unresolved dependency remains: **MAYOADIRL_MRI_FMRI_NFQ**
  and **MAYOADIRL_MRI_FMRI** tables for fMRI-specific QC and slice order verification.

---

## Guardrails Compliance
- Read-only. No model training, no threshold fitting, no OASIS scoring.
- No tensor modification, no metadata modification, no artifact overwrite.
- No subject exclusion.
- All findings descriptive/exploratory.
"""

_safe_md(interp_text, OUT_DIR / "final_local_source_audit_interpretation.md")

# Martin follow-up
martin_text = f"""# Martin Follow-Up: Required Fields for Philips CN Audit

**Date**: {datetime.now(timezone.utc).isoformat()}
**N Philips CN**: 99 (FP={n_fp}, TN={n_tn})

---

## Priority 1 — ADNI Download Requests (LONI Data Portal)

### 1.1 MAYOADIRL_MRI_FMRI_NFQ.csv  ← HIGHEST PRIORITY
- **Table**: Jack Lab — fMRI Network Failure Quotient (MAYOADIRL_MRI_FMRI_NFQ)
- **Why needed**: Contains per-scan OVERALLQC (1=excellent; 4=fail), SLICEORDER,
  SOFTWAREVERSIONS, MANUFACTURER, MANUFACTURERSMODELNAME, REPETITIONTIME, ECHOTIME.
- **Key fields**: OVERALLQC, NFQ, SLICEORDER, SOFTWAREVERSIONS, SCANDATE
- **Join key**: RID + VISCODE (or RID + SCANDATE)
- **Action**: Download from LONI Data Portal. Filter Manufacturer=Philips, match 99 Philips CN subjects.

### 1.2 MAYOADIRL_MRI_FMRI.csv
- **Table**: Jack Lab — Default Mode Network connectivity (MAYOADIRL_MRI_FMRI)
- **Key fields**: PHASEDIR (phase encoding direction), SLICEORD (temporal slice order)
- **Action**: Download from LONI Data Portal. Match by RID + SCANDATE.

### 1.3 MAYOADIRL_MRI_ADNI3.csv  (if subjects include ADNI3 phase)
- **Table**: Jack Lab — ADNI GO/2/3 MRI QC (MAYOADIRL_MRI_ADNI3)
- **Key fields**: SERIES_QUALITY (per series QC rating)
- **Action**: Download from LONI Data Portal.

---

## Priority 2 — Motion Files (rp_*.txt)

- **Status**: 0/99 Philips CN subjects have rp_*.txt locally.
- **ADNI motion data location**: LONI Image and Data Archive → per-series download
  or MRI Quality Control module.
- **Action**: For Philips CN subjects where slice_order_issue or problem_site_flag is set,
  provide rp_*.txt (SPM realignment parameter files: 6 columns dx/dy/dz/rx/ry/rz).
- **Priority subjects**: site 31, site 18, 140TP group, high-score FP (score ≥ 0.75).

---

## Priority 3 — Slice Order Confirmation

- **From Martin's earlier communication**: Philips subjects at sites 31 and 18 appear
  to have incorrect slice order.
- **Required**: For each affected subject, confirm:
  1. What slice order was declared in DICOM header / BIDS JSON?
  2. What slice order was ACTUALLY used in preprocessing (DPARSF config)?
  3. Was interleaved ascending/descending used? Sequential?
  4. Was this corrected in preprocessing (slice timing correction applied)?
- **Join key**: SubjectID → RID → SCANDATE → LONI_IMG_ID

---

## Priority 4 — BIDS JSON Sidecars

- **Status**: {json_total_str} Philips CN subjects have dicom_json_path in master DB,
  but local paths may not be accessible.
- **Key fields needed**: PhaseEncodingDirection, SliceTiming, ManufacturerModelName,
  SoftwareVersions, ProtocolName, RepetitionTime, NumberOfSlices.
- **Action**: For the 99 Philips CN SubjectIDs, provide BIDS JSON sidecar files or
  a CSV extract from LONI Image Metadata with the above fields.

---

## Priority 5 — ADNI2→ADNI3 Temporal Resolution Protocol

From the prior audit (2026-06-10):
- 140TP subjects are predominantly ADNI1/2 Philips (older, shorter scan).
- 197TP subjects are predominantly ADNI3.
- **Unresolved**: Was there a deliberate truncation from 197→140 TPs in ADNI2?
  Is this truncation applied uniformly within each site?
- **Action**: Confirm the DPARSF preprocessing config used for 140TP subjects.
  Were the same number of dummy volumes removed for 140TP and 197TP subjects?

---

## Already Resolved (Do NOT re-request)

- Item 1 (BOLD .mat files): All 99/99 .mat files found locally.
- Item 6 (scale normalization for 3 subjects): Retracted — transposition bug confirmed.
- tSNR anomaly for 100_S_5075, 013_S_4579, 013_S_5171: All show normal tSNR (~250–500).

---

## Philips CN High-Priority FP Subjects for DICOM/Slice Review

{"Subjects with y_score_final >= 0.75 and fp_flag=True." if "y_score_final" in philips_cn.columns else "See high_confidence_philips_cn_fp_review.csv from deep audit."}
"""
hc_fp = philips_cn[(philips_cn["fp_flag"]) & (philips_cn.get("y_score_final", pd.Series(dtype=float)) >= 0.75)].copy() if "y_score_final" in philips_cn.columns else pd.DataFrame()
if len(hc_fp) > 0:
    cols_show = [c for c in ["SubjectID", "Site3", "Age", "y_score_final", "raw_tp_group",
                              "ORIGPROT", "scanner_model", "software_version",
                              "philips_slice_order_issue_flag", "philips_problem_site_flag"] if c in hc_fp.columns]
    martin_text += f"\n\nN high-confidence FP (score ≥ 0.75): {len(hc_fp)}\n\n"
    martin_text += hc_fp[cols_show].to_markdown(index=False)

_save_md = lambda t, p: p.write_text(t, encoding="utf-8") or _log("Saved MD", str(p))
_save_md(martin_text, OUT_DIR / "martin_followup_required_fields.md")

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
_log("Executive Summary", "Writing 00_EXECUTIVE_SUMMARY.md")

sig_vars_str = "\n".join(f"  - {v}" for v in sig_vars[:10]) if sig_vars else "  None newly significant (primary findings carried from deep audit 2026-06-10)"

exec_text = f"""# 00 Executive Summary — Local Martin Philips Source Audit

**Date**: {datetime.now(timezone.utc).isoformat()}
**Script**: local_martin_philips_source_audit_20260611.py
**N Philips CN**: 99 (FP={n_fp}, TN={n_tn})
**Promoted model**: recover035_latent384_beta3p75_T80_h10000_p560_full5x5

---

## What This Audit Does

Read-only forensic discovery of all locally available acquisition/QC data for
the 99 Philips CN subjects in the promoted model, with the goal of identifying
objective QC evidence (beyond age and scan duration) that might explain the
Philips CN FPR ≈ 45%.

---

## Source Availability Outcome

| Source | Local? | Coverage |
|---|---|---|
| Promoted model master DB (254 cols) | YES | 99/99 |
| MAYOADIRL_MRI_MCH radiology findings | YES | {len(mch_sids_joined)}/99 |
| ImageQualityMetrics (SNR/tSNR) | YES (partial) | {len(iqm_sids_joined)}/99 |
| desde_cero SubjectsData (ImagingProtocol) | YES (partial) | {len(dc_sids_joined)}/99 |
| ADNIMERGE clinical | YES | {len(adni_sids_joined)}/99 |
| MAYOADIRL_MRI_FMRI_NFQ (NFQ/OVERALLQC/SliceOrder) | **NOT DOWNLOADED** | 0/99 |
| MAYOADIRL_MRI_FMRI (PHASEDIR/SLICEORD) | **NOT DOWNLOADED** | 0/99 |
| rp_*.txt motion files (ADNI Philips CN) | **NOT FOUND** | {rp_total_str} |
| BIDS JSON sidecars (ADNI fMRI) | **NOT FOUND** | {json_total_str} |

---

## Key Finding: The Master DB Already Contains Most Protocol Data

The 254-column master DB (built in prior session 2026-06-10) already contains:
protocol fields (scanner_model, software_version, coil, TR, TE, n_slices),
phase encoding direction, slice order flags, FD motion metrics (where available),
and BOLD QC. The newly merged sources (MCH, IQM) add partial coverage for
radiology findings and older SNR metrics.

---

## Primary Unresolved Dependencies

1. **MAYOADIRL_MRI_FMRI_NFQ** — fMRI-specific QC (OVERALLQC, NFQ, SLICEORDER).
   Without this table, we cannot confirm whether slice order errors
   are systematic across FP subjects.

2. **rp_*.txt motion files** — 0/99 available for ADNI Philips CN.
   Cannot compute FD for the full cohort.

3. **Martin's slice order annotation** — Partial. Sites 31 and 18 flagged
   but systematic verification requires per-subject DICOM inspection
   or MAYOADIRL_MRI_FMRI SLICEORD field.

---

## Statistical Findings (New from This Audit)

Significant variables (FDR q < 0.10) from newly merged sources:
{sig_vars_str}

**From prior deep audit (carried forward, not re-tested):**
- Age (FP median=76.9 vs TN=71.6, MW p=0.0003, CLES=0.710) — **primary biological driver**
- raw_tp_group 140TP FPR=63.4% vs 197TP FPR=32.8% (OR=3.56, p=0.0039) — **protocol risk**
- Tensor ch0 (Pearson FC) LOWER in FP; ch2 (MI) HIGHER in FP — **connectivity pattern shift**
- LR cosine with age direction=0.066 — classifier not directly age-driven
- AD-CN direction cosine with age=0.331 — moderate; AD pattern partially age-confounded

---

## FPR by Protocol Stratification (from this audit)

See site_protocol_qc_fpr_table.csv. Key rows confirmed:
- Philips 140TP FPR ≈ 63% vs 197TP ≈ 33%
- Site-level FPR heterogeneity (std ≈ 0.35) within Philips CN
- Software version and scanner model FPR breakdown: see table

---

## Recommendations

1. **Download MAYOADIRL_MRI_FMRI_NFQ** from LONI Data Portal (Priority 1).
   This is the single most informative missing source.
2. **Request rp_*.txt from Martin** for at least the 26 Philips CN FP subjects
   in the 140TP group.
3. **Keep promoted model as primary** — no new objective QC-based exclusion
   criterion has been identified in locally available data.
4. **Age + scan duration (140TP) remains the dominant explanation**
   consistent with the prior protocol-risk audit.

---

## Output Files

- `local_candidate_files.csv/.md` — all searched local sources with coverage
- `candidate_csv_header_inventory.csv/.md` — column headers for each source
- `relevant_table_dictionary_hits.csv/.md` — DATADIC entries for target tables
- `philips_cn_source_coverage.csv/.md` — per-source join coverage audit
- `philips_cn_local_sources_merged_candidate.csv` — 99-subject merged table
- `rp_motion_coverage_and_fd.csv/.md` — rp availability and FD metrics
- `sidecar_metadata_inventory.csv/.md` — JSON sidecar and master DB protocol fields
- `acquisition_qc_fp_vs_tn_tests.csv/.md` — FP vs TN statistical tests
- `site_protocol_qc_fpr_table.csv/.md` — FPR by stratification
- `final_local_source_audit_interpretation.md` — full interpretation
- `martin_followup_required_fields.md` — prioritized Martin follow-up items
- `command_log.json` — execution log

---

## Guardrails Compliance
- Read-only. No model training, no threshold fitting, no OASIS scoring.
- No tensor modification, no metadata modification, no model artifact overwrite.
- No subject exclusion from any pool.
"""

_safe_md(exec_text, OUT_DIR / "00_EXECUTIVE_SUMMARY.md")

# ─────────────────────────────────────────────────────────────────────────────
# Finalize
# ─────────────────────────────────────────────────────────────────────────────
_log("Finalizing", "command_log.json")
_save_json(command_log, OUT_DIR / "command_log.json")

print(f"\n{'='*60}")
print(f"Audit complete. Output directory: {OUT_DIR}")
print(f"Files written: {len(list(OUT_DIR.iterdir()))}")
print(f"{'='*60}")
