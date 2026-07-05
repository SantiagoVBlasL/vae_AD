"""
Pre-training QC audit for ADNI v5 DPARSF-10000 no-Python-bandpass tensor.

Checks tensor/metadata alignment, demographics completeness, channel fallback
status, and produces a training-ready manifest with go/no-go recommendation.

Usage:
    python qc_v5_dparsf10000_no_pybandpass_pretraining.py [--output-root DIR]

Outputs (default: results/revision_bspc_2026/adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc/):
    README.md
    tensor_metadata_alignment.csv
    missing_demographics_subjects.csv
    unknown_diagnosis_subjects.csv
    supervised_subject_counts.csv
    channel_fallback_subjects.csv
    training_ready_manifest.csv
"""

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TENSOR_ROOT = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
)
_GLOBAL_TENSOR = (
    _TENSOR_ROOT
    / "subject_tensors"
    / "GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
_METADATA = (
    _TENSOR_ROOT / "subject_metadata_v5_dparsf10000_no_pybandpass.csv"
)
_MANIFEST = (
    _TENSOR_ROOT / "subject_manifest_v5_dparsf10000_no_pybandpass.csv"
)
_EXTRACTION_QC = _TENSOR_ROOT / "subject_tensors" / "full_extraction_qc.csv"
_CHANNEL_QC = _TENSOR_ROOT / "subject_tensors" / "full_channel_qc.csv"
_EXCLUDED = _TENSOR_ROOT / "excluded_subjects_v5.csv"
_DUP_RESOLUTION = _TENSOR_ROOT / "duplicate_resolution_v5.csv"

_DEFAULT_OUT = (
    _REPO_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_expanded_v5_dparsf10000_no_pybandpass_pretraining_qc"
)

# Expected tensor properties
EXPECTED_CHANNELS = 7
EXPECTED_ROIS = 131
EXPECTED_TR = 3.0
EXPECTED_TARGET_LEN = 140
EXPECTED_CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]

# channel calc_status values that indicate a non-standard (fallback) computation
_FALLBACK_PATTERNS = ["fallback_mst", "fallback"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUT,
        help="Directory to write QC outputs (default: %(default)s)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_path(p: Path, label: str) -> bool:
    if not p.exists():
        print(f"  [ERROR] {label} not found: {p}", file=sys.stderr)
        return False
    return True


def _group_counts(series: pd.Series) -> dict:
    return series.value_counts(dropna=False).to_dict()


# ---------------------------------------------------------------------------
# 1. Load global tensor
# ---------------------------------------------------------------------------

def load_global_tensor(path: Path) -> dict:
    print(f"Loading global tensor: {path}")
    d = np.load(path, allow_pickle=False)
    tensor = d["global_tensor_data"]
    ids = list(d["subject_ids"].tolist())
    return {
        "tensor": tensor,
        "subject_ids": ids,
        "channel_names": list(d["channel_names"].tolist()),
        "rois_count": int(d["rois_count"]),
        "target_len_ts": int(d["target_len_ts"]),
        "tr_seconds": float(d["tr_seconds"]),
        "python_bandpass_applied": bool(d["python_bandpass_applied"]),
        "roi_order_name": str(d["roi_order_name"]),
    }


# ---------------------------------------------------------------------------
# 2. Confirm tensor properties
# ---------------------------------------------------------------------------

def audit_tensor_properties(tinfo: dict) -> tuple[bool, list[str]]:
    """Return (all_ok, list_of_issues)."""
    t = tinfo["tensor"]
    issues = []

    n_subj, n_ch, n_roi1, n_roi2 = t.shape
    if n_ch != EXPECTED_CHANNELS:
        issues.append(f"channels: got {n_ch}, expected {EXPECTED_CHANNELS}")
    if n_roi1 != EXPECTED_ROIS or n_roi2 != EXPECTED_ROIS:
        issues.append(f"roi dims: got ({n_roi1},{n_roi2}), expected ({EXPECTED_ROIS},{EXPECTED_ROIS})")
    if t.dtype != np.float32:
        issues.append(f"dtype: got {t.dtype}, expected float32")
    nan_count = int(np.isnan(t).sum())
    if nan_count > 0:
        issues.append(f"NaN count: {nan_count} (expected 0)")
    if not np.isfinite(t).all():
        issues.append("tensor contains non-finite values (inf/-inf)")
    if tinfo["tr_seconds"] != EXPECTED_TR:
        issues.append(f"TR: got {tinfo['tr_seconds']}, expected {EXPECTED_TR}")
    if tinfo["target_len_ts"] != EXPECTED_TARGET_LEN:
        issues.append(f"target_len: got {tinfo['target_len_ts']}, expected {EXPECTED_TARGET_LEN}")
    if tinfo["python_bandpass_applied"]:
        issues.append("python_bandpass_applied is True (expected False)")
    if tinfo["channel_names"] != EXPECTED_CHANNEL_NAMES:
        issues.append(f"channel_names mismatch: got {tinfo['channel_names']}")

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# 3. Alignment
# ---------------------------------------------------------------------------

def audit_alignment(tensor_ids: list, meta: pd.DataFrame) -> dict:
    tensor_set = set(tensor_ids)
    meta_set = set(meta["SubjectID"].tolist())

    in_tensor_not_meta = sorted(tensor_set - meta_set)
    in_meta_not_tensor = sorted(meta_set - tensor_set)
    in_both = sorted(tensor_set & meta_set)

    return {
        "n_tensor": len(tensor_set),
        "n_meta": len(meta_set),
        "n_in_both": len(in_both),
        "in_tensor_not_meta": in_tensor_not_meta,
        "in_meta_not_tensor": in_meta_not_tensor,
        "in_both": in_both,
    }


# ---------------------------------------------------------------------------
# 4. Demographics audit
# ---------------------------------------------------------------------------

def audit_demographics(meta: pd.DataFrame, tensor_ids: list) -> dict:
    tensor_set = set(tensor_ids)

    nan_diag = meta[meta["ResearchGroup_Mapped"].isna()].copy()
    nan_age = meta[meta["Age"].isna()].copy()
    nan_sex = meta[meta["Sex"].isna()].copy()
    nan_any_demo = meta[meta["Age"].isna() | meta["Sex"].isna()].copy()

    for df in [nan_diag, nan_age, nan_sex, nan_any_demo]:
        df["in_tensor"] = df["SubjectID"].isin(tensor_set)

    return {
        "nan_diagnosis": nan_diag,
        "nan_age": nan_age,
        "nan_sex": nan_sex,
        "nan_any_demographics": nan_any_demo,
    }


# ---------------------------------------------------------------------------
# 5. Subject counts and training cohorts
# ---------------------------------------------------------------------------

def compute_cohort_counts(meta: pd.DataFrame, tensor_ids: list) -> dict:
    tensor_set = set(tensor_ids)

    # Only subjects actually in the tensor
    in_tensor = meta[meta["SubjectID"].isin(tensor_set)].copy()
    in_tensor["in_tensor"] = True

    group_counts = _group_counts(in_tensor["ResearchGroup_Mapped"])

    excl_flag = in_tensor["exclude_from_supervised"].fillna(False).astype(bool)
    has_demo = in_tensor["Age"].notna() & in_tensor["Sex"].notna()
    has_diag = in_tensor["ResearchGroup_Mapped"].notna()

    cn_ad = in_tensor[in_tensor["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    supervised_all = cn_ad[~excl_flag.reindex(cn_ad.index, fill_value=False)]
    supervised_with_demo = supervised_all[
        supervised_all["Age"].notna() & supervised_all["Sex"].notna()
    ]

    # VAE: all in tensor (no diagnosis required), but flag excluded ones
    vae_all = in_tensor
    vae_usable = in_tensor  # all 495 are usable for VAE unless there's a specific exclusion

    # Per-group supervised counts
    sup_counts = supervised_all["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
    sup_demo_counts = supervised_with_demo["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()

    return {
        "total_in_tensor": len(in_tensor),
        "group_counts": group_counts,
        "n_supervised_all": len(supervised_all),
        "supervised_group_counts": sup_counts,
        "n_supervised_with_demo": len(supervised_with_demo),
        "supervised_with_demo_group_counts": sup_demo_counts,
        "n_vae_all": len(vae_all),
        "subjects_in_tensor_excluded_from_supervised": sorted(
            cn_ad[excl_flag.reindex(cn_ad.index, fill_value=False)]["SubjectID"].tolist()
        ),
        "subjects_in_tensor_missing_demo": sorted(
            in_tensor[~has_demo]["SubjectID"].tolist()
        ),
        "subjects_in_tensor_missing_diagnosis": sorted(
            in_tensor[~has_diag]["SubjectID"].tolist()
        ),
    }


# ---------------------------------------------------------------------------
# 6. Channel fallback audit
# ---------------------------------------------------------------------------

def audit_channel_fallback(channel_qc: pd.DataFrame) -> pd.DataFrame:
    is_fallback = channel_qc["calc_status"].apply(
        lambda s: any(p in str(s) for p in _FALLBACK_PATTERNS)
    )
    fallback_rows = channel_qc[is_fallback].copy()
    return fallback_rows[["SubjectID", "channel_name", "calc_status",
                           "raw_nan_count", "scaled_nan_count"]]


# ---------------------------------------------------------------------------
# 7. Training-ready manifest
# ---------------------------------------------------------------------------

def build_training_manifest(
    meta: pd.DataFrame,
    tensor_ids: list,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    tensor_set = set(tensor_ids)

    base_cols = [
        "SubjectID", "ResearchGroup_Mapped", "Age", "Sex",
        "Manufacturer", "Site3", "ImageID", "Visit",
        "metadata_source", "exclude_from_supervised",
    ]
    mf_cols = ["SubjectID", "scale_label", "finite_fraction",
               "preprocessing_source", "python_bandpass_applied",
               "TR", "target_len"]

    df = meta[[c for c in base_cols if c in meta.columns]].copy()
    df["in_tensor"] = df["SubjectID"].isin(tensor_set)

    mf_sub = manifest[[c for c in mf_cols if c in manifest.columns]].drop_duplicates("SubjectID")
    df = df.merge(mf_sub, on="SubjectID", how="left")

    has_demo = df["Age"].notna() & df["Sex"].notna()
    has_diag = df["ResearchGroup_Mapped"].notna()
    excl_flag = df["exclude_from_supervised"].fillna(False).astype(bool)
    is_cn_ad = df["ResearchGroup_Mapped"].isin(["CN", "AD"])

    df["use_for_vae"] = df["in_tensor"]
    df["use_for_supervised"] = (
        df["in_tensor"] & is_cn_ad & ~excl_flag
    )
    df["use_for_supervised_with_demo"] = (
        df["in_tensor"] & is_cn_ad & ~excl_flag & has_demo
    )

    df = df.sort_values(["ResearchGroup_Mapped", "SubjectID"], na_position="last")
    return df


# ---------------------------------------------------------------------------
# 8. Training readiness decision
# ---------------------------------------------------------------------------

def evaluate_readiness(
    tensor_ok: bool,
    tensor_issues: list,
    alignment: dict,
    cohort: dict,
    fallback_df: pd.DataFrame,
) -> tuple[bool, list[str], list[str]]:
    blocking = []
    warnings = []

    if not tensor_ok:
        blocking.extend([f"TENSOR: {i}" for i in tensor_issues])

    if alignment["in_tensor_not_meta"]:
        blocking.append(
            f"Subjects in tensor but NOT in metadata: {alignment['in_tensor_not_meta']}"
        )

    if cohort["total_in_tensor"] == 0:
        blocking.append("No subjects in tensor.")

    if cohort["n_supervised_with_demo"] < 100:
        blocking.append(
            f"Supervised pool with demographics too small: {cohort['n_supervised_with_demo']}"
        )

    # Warnings (non-blocking)
    if alignment["in_meta_not_tensor"]:
        warnings.append(
            f"Subjects in metadata but NOT in tensor (expected for excluded): "
            f"{alignment['in_meta_not_tensor']}"
        )

    if cohort["subjects_in_tensor_missing_demo"]:
        warnings.append(
            f"Subjects in tensor with missing Age/Sex: "
            f"{cohort['subjects_in_tensor_missing_demo']}"
        )

    if cohort["subjects_in_tensor_missing_diagnosis"]:
        warnings.append(
            f"Subjects in tensor with unknown diagnosis (usable for VAE only): "
            f"{cohort['subjects_in_tensor_missing_diagnosis']}"
        )

    hard_fallbacks = fallback_df[
        fallback_df["calc_status"].str.contains("fallback_mst", na=False)
    ]
    if len(hard_fallbacks) > 0:
        warnings.append(
            f"OMST hard fallback (MST) in {hard_fallbacks['SubjectID'].nunique()} subject(s): "
            f"{sorted(hard_fallbacks['SubjectID'].unique().tolist())} — "
            "tensor entry is valid (MST is a legitimate sparse connectivity estimate) "
            "but these subjects should be noted in methods."
        )

    ready = len(blocking) == 0
    return ready, blocking, warnings


# ---------------------------------------------------------------------------
# 9. README
# ---------------------------------------------------------------------------

def write_readme(
    out_dir: Path,
    tensor_info: dict,
    tensor_ok: bool,
    tensor_issues: list,
    alignment: dict,
    cohort: dict,
    demo_audit: dict,
    fallback_df: pd.DataFrame,
    ready: bool,
    blocking: list,
    warnings_list: list,
    meta_path: Path,
    manifest_path: Path,
    run_ts: str,
):
    t = tensor_info["tensor"]
    n_subj = t.shape[0]

    hard_fb = fallback_df[fallback_df["calc_status"].str.contains("fallback_mst", na=False)]
    hard_fb_subs = sorted(hard_fb["SubjectID"].unique().tolist())

    lines = [
        "# ADNI v5 DPARSF-10000 No-Python-Bandpass — Pre-Training QC",
        "",
        f"Generated: {run_ts}",
        "",
        "## 1. Tensor Properties",
        "",
        f"- Path: `{_GLOBAL_TENSOR}`",
        f"- Shape: {t.shape}  (subjects × channels × ROIs × ROIs)",
        f"- dtype: {t.dtype}",
        f"- Finite: {np.isfinite(t).all()}",
        f"- NaNs: {int(np.isnan(t).sum())}",
        f"- Channels ({len(tensor_info['channel_names'])}):",
    ]
    for ch in tensor_info["channel_names"]:
        lines.append(f"  - {ch}")
    lines += [
        f"- ROIs: {tensor_info['rois_count']}",
        f"- TR: {tensor_info['tr_seconds']} s",
        f"- Target length: {tensor_info['target_len_ts']}",
        f"- Python bandpass applied: {tensor_info['python_bandpass_applied']}",
        f"- ROI order: {tensor_info['roi_order_name']}",
        f"- Tensor property checks: {'PASS' if tensor_ok else 'FAIL'}",
    ]
    if not tensor_ok:
        for iss in tensor_issues:
            lines.append(f"  - ISSUE: {iss}")

    lines += [
        "",
        "## 2. Tensor / Metadata Alignment",
        "",
        f"- Subjects in tensor: {alignment['n_tensor']}",
        f"- Subjects in metadata: {alignment['n_meta']}",
        f"- Subjects in both: {alignment['n_in_both']}",
        f"- In tensor but NOT metadata: {alignment['in_tensor_not_meta'] or 'none'}",
        f"- In metadata but NOT tensor: {alignment['in_meta_not_tensor'] or 'none'}",
        "  (expected: excluded subjects that passed manifest QC but failed extraction criteria)",
        "",
        "## 3. Diagnosis Groups (in tensor)",
        "",
    ]
    for grp, cnt in sorted(cohort["group_counts"].items(), key=lambda x: str(x[0])):
        label = str(grp) if grp is not None else "NaN/Unknown"
        lines.append(f"- {label}: {cnt}")

    lines += [
        "",
        "## 4. Demographics Completeness",
        "",
        f"- Subjects with NaN Age or Sex (in tensor): "
        f"{cohort['subjects_in_tensor_missing_demo'] or 'none'}",
        f"- Subjects with NaN diagnosis (in tensor): "
        f"{cohort['subjects_in_tensor_missing_diagnosis'] or 'none'}",
        "",
        "### Notes",
        "- `035_S_6927` (AD): **in tensor**, Age and Sex are NaN — no demographics record found in any "
        "ADNI source. Diagnosis confirmed via `AD_fMRI_4_28_2026.csv` and `idaSearch_4_03_2026.csv`. "
        "Usable for VAE and for supervised (no-demographics filter). "
        "Excluded from supervised-with-demographics cohort.",
        "- `128_S_2002` (NaN diagnosis): **in tensor**, `exclude_from_supervised=True`. "
        "Signal anomaly (96.9% near-zero values, scale_label=unknown); "
        "OMST channel fell back to MST. Excluded from supervised training. "
        "Usable for VAE in principle but anomalous signal is a concern.",
        "- `114_S_6039` (AD): **not in tensor** — excluded from extraction "
        "(see `excluded_subjects_v5.csv`). Does not appear in tensor or training cohorts.",
        "",
        "## 5. Training Cohort Summary",
        "",
        f"- Total subjects in tensor: {cohort['total_in_tensor']}",
        f"- Usable for VAE (diagnosis-agnostic): {cohort['n_vae_all']}",
        f"  (includes all 495 tensor subjects regardless of diagnosis or demographics)",
        f"- Supervised AD/CN — all (no demographics filter): {cohort['n_supervised_all']}",
    ]
    for grp, cnt in sorted(cohort["supervised_group_counts"].items(), key=lambda x: str(x[0])):
        lines.append(f"  - {grp}: {cnt}")
    lines += [
        f"- Supervised AD/CN — with complete Age+Sex: {cohort['n_supervised_with_demo']}",
    ]
    for grp, cnt in sorted(cohort["supervised_with_demo_group_counts"].items(), key=lambda x: str(x[0])):
        lines.append(f"  - {grp}: {cnt}")
    lines += [
        f"- Excluded from supervised (flag): "
        f"{cohort['subjects_in_tensor_excluded_from_supervised'] or 'none'}",
        "",
        "## 6. Channel Fallback",
        "",
        f"- Subjects with OMST hard fallback to MST: "
        f"{hard_fb_subs if hard_fb_subs else 'none'}",
        "  - `128_S_2002` (in tensor, exclude_from_supervised=True) had `dyconnmap_failed_fallback_mst` "
        "for Pearson_OMST_GCE_Signed_Weighted channel.",
        "  - MST is a valid sparse connectivity estimate; the tensor entry is numerically "
        "sound (NaN=0 for that subject). This subject is excluded from supervised training "
        "due to unresolvable diagnosis, so it does not affect the supervised classifier.",
        "  - DistanceCorr uses `dcor` library (calc_status = `dcor.distance_correlation`) "
        "for all 495 subjects — this is normal; not a fallback.",
        "",
        "## 7. Files in This Directory",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `tensor_metadata_alignment.csv` | All metadata subjects with in_tensor flag and alignment status |",
        "| `missing_demographics_subjects.csv` | Subjects missing Age or Sex |",
        "| `unknown_diagnosis_subjects.csv` | Subjects with NaN ResearchGroup_Mapped |",
        "| `supervised_subject_counts.csv` | Count summary for training cohorts |",
        "| `channel_fallback_subjects.csv` | Channel rows with non-standard calc_status |",
        "| `training_ready_manifest.csv` | Full manifest with use_for_vae / use_for_supervised flags |",
        "",
        "## 8. Training Readiness",
        "",
        f"**ready_for_baseline_training: {'YES' if ready else 'NO'}**",
        "",
    ]
    if blocking:
        lines.append("### Blocking Issues")
        for b in blocking:
            lines.append(f"- {b}")
        lines.append("")
    if warnings_list:
        lines.append("### Warnings (non-blocking)")
        for w in warnings_list:
            lines.append(f"- {w}")
        lines.append("")

    lines += [
        "### Recommendations",
        "",
        "- **Subjects to exclude from supervised**: `128_S_2002` "
        "(in tensor; `exclude_from_supervised=True`, NaN diagnosis) and `114_S_6039` (not in tensor)",
        "- **Subjects to exclude from supervised-with-demographics**: "
        "`035_S_6927` (AD, NaN Age/Sex) and `128_S_2002`",
        "- **Subjects to exclude from VAE if needed**: `128_S_2002` — in tensor but "
        "severe signal anomaly (96.9% near-zero, OMST fallback). "
        "`114_S_6039` is already absent from tensor.",
        f"- **Final metadata path for training**: `{meta_path}`",
        f"- **Final manifest path for training**: `{manifest_path}`",
        "- **Use `training_ready_manifest.csv`** (in this directory) as the definitive "
        "subject list with `use_for_vae` and `use_for_supervised` flags.",
    ]

    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"  Wrote README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir: Path = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=== ADNI v5 DPARSF-10000 No-Python-Bandpass Pre-Training QC ===")
    print(f"Output dir: {out_dir}")
    print()

    # Check all inputs exist
    ok = True
    for path, label in [
        (_GLOBAL_TENSOR, "global tensor"),
        (_METADATA, "metadata"),
        (_MANIFEST, "manifest"),
        (_EXTRACTION_QC, "extraction QC"),
        (_CHANNEL_QC, "channel QC"),
    ]:
        ok = _check_path(path, label) and ok
    if not ok:
        print("One or more required files missing. Aborting.", file=sys.stderr)
        sys.exit(1)

    # ---- Load ----
    print("[1] Loading global tensor ...")
    tensor_info = load_global_tensor(_GLOBAL_TENSOR)
    tensor_ids = tensor_info["subject_ids"]
    print(f"    shape={tensor_info['tensor'].shape}, dtype={tensor_info['tensor'].dtype}, "
          f"n_subjects={len(tensor_ids)}")

    print("[2] Loading metadata, manifest, QC files ...")
    meta = pd.read_csv(_METADATA)
    manifest = pd.read_csv(_MANIFEST)
    extraction_qc = pd.read_csv(_EXTRACTION_QC)
    channel_qc = pd.read_csv(_CHANNEL_QC)
    print(f"    metadata: {meta.shape}  manifest: {manifest.shape}  "
          f"extraction_qc: {extraction_qc.shape}  channel_qc: {channel_qc.shape}")

    # ---- Audits ----
    print("[3] Auditing tensor properties ...")
    tensor_ok, tensor_issues = audit_tensor_properties(tensor_info)
    status_str = "PASS" if tensor_ok else f"FAIL ({tensor_issues})"
    print(f"    {status_str}")

    print("[4] Auditing tensor / metadata alignment ...")
    alignment = audit_alignment(tensor_ids, meta)
    print(f"    n_tensor={alignment['n_tensor']}  n_meta={alignment['n_meta']}  "
          f"in_both={alignment['n_in_both']}")
    if alignment["in_tensor_not_meta"]:
        print(f"    [WARN] in tensor not meta: {alignment['in_tensor_not_meta']}")
    if alignment["in_meta_not_tensor"]:
        print(f"    [info] in meta not tensor: {alignment['in_meta_not_tensor']}")

    print("[5] Auditing demographics ...")
    demo_audit = audit_demographics(meta, tensor_ids)
    print(f"    NaN diagnosis: {len(demo_audit['nan_diagnosis'])} subject(s)")
    print(f"    NaN Age or Sex: {len(demo_audit['nan_any_demographics'])} subject(s)")

    print("[6] Computing cohort counts ...")
    cohort = compute_cohort_counts(meta, tensor_ids)
    print(f"    total in tensor: {cohort['total_in_tensor']}")
    print(f"    groups: {cohort['group_counts']}")
    print(f"    supervised AD/CN (all): {cohort['n_supervised_all']}  "
          f"{cohort['supervised_group_counts']}")
    print(f"    supervised AD/CN (with demo): {cohort['n_supervised_with_demo']}  "
          f"{cohort['supervised_with_demo_group_counts']}")

    print("[7] Auditing channel fallbacks ...")
    fallback_df = audit_channel_fallback(channel_qc)
    print(f"    non-standard channel rows: {len(fallback_df)}")
    for _, row in fallback_df.iterrows():
        print(f"      {row['SubjectID']} | {row['channel_name']} | {row['calc_status']}")

    print("[8] Evaluating training readiness ...")
    ready, blocking, warn_list = evaluate_readiness(
        tensor_ok, tensor_issues, alignment, cohort, fallback_df
    )
    print(f"    ready_for_baseline_training: {'YES' if ready else 'NO'}")
    if blocking:
        print(f"    BLOCKING: {blocking}")
    for w in warn_list:
        print(f"    WARN: {w}")

    # ---- Build training manifest ----
    print("[9] Building training-ready manifest ...")
    training_manifest = build_training_manifest(meta, tensor_ids, manifest)

    # ---- Write outputs ----
    print("[10] Writing outputs ...")

    # tensor_metadata_alignment.csv
    align_df = meta.copy()
    align_df["in_tensor"] = align_df["SubjectID"].isin(set(tensor_ids))
    align_df["alignment_status"] = align_df["in_tensor"].map(
        {True: "in_both", False: "meta_only"}
    )
    align_df.to_csv(out_dir / "tensor_metadata_alignment.csv", index=False)
    print(f"  Wrote tensor_metadata_alignment.csv ({len(align_df)} rows)")

    # missing_demographics_subjects.csv
    demo_audit["nan_any_demographics"].to_csv(
        out_dir / "missing_demographics_subjects.csv", index=False
    )
    print(f"  Wrote missing_demographics_subjects.csv "
          f"({len(demo_audit['nan_any_demographics'])} rows)")

    # unknown_diagnosis_subjects.csv
    demo_audit["nan_diagnosis"].to_csv(
        out_dir / "unknown_diagnosis_subjects.csv", index=False
    )
    print(f"  Wrote unknown_diagnosis_subjects.csv "
          f"({len(demo_audit['nan_diagnosis'])} rows)")

    # supervised_subject_counts.csv
    count_rows = []
    count_rows.append({
        "cohort": "total_in_tensor",
        "n_total": cohort["total_in_tensor"],
        "n_CN": cohort["group_counts"].get("CN", 0),
        "n_AD": cohort["group_counts"].get("AD", 0),
        "n_MCI": cohort["group_counts"].get("MCI", 0),
        "n_Unknown": cohort["group_counts"].get(float("nan"), 0)
        + cohort["group_counts"].get(None, 0),
        "note": "All subjects extracted into global tensor",
    })
    count_rows.append({
        "cohort": "vae_diagnosis_agnostic",
        "n_total": cohort["n_vae_all"],
        "n_CN": cohort["group_counts"].get("CN", 0),
        "n_AD": cohort["group_counts"].get("AD", 0),
        "n_MCI": cohort["group_counts"].get("MCI", 0),
        "n_Unknown": cohort["group_counts"].get(float("nan"), 0)
        + cohort["group_counts"].get(None, 0),
        "note": "All tensor subjects usable; no diagnosis or demographics required",
    })
    count_rows.append({
        "cohort": "supervised_CN_AD_all",
        "n_total": cohort["n_supervised_all"],
        "n_CN": cohort["supervised_group_counts"].get("CN", 0),
        "n_AD": cohort["supervised_group_counts"].get("AD", 0),
        "n_MCI": 0,
        "n_Unknown": 0,
        "note": "CN+AD, exclude_from_supervised=False; no demographics filter",
    })
    count_rows.append({
        "cohort": "supervised_CN_AD_with_demographics",
        "n_total": cohort["n_supervised_with_demo"],
        "n_CN": cohort["supervised_with_demo_group_counts"].get("CN", 0),
        "n_AD": cohort["supervised_with_demo_group_counts"].get("AD", 0),
        "n_MCI": 0,
        "n_Unknown": 0,
        "note": "CN+AD with complete Age+Sex; recommended for age/sex-adjusted models",
    })
    pd.DataFrame(count_rows).to_csv(
        out_dir / "supervised_subject_counts.csv", index=False
    )
    print(f"  Wrote supervised_subject_counts.csv ({len(count_rows)} rows)")

    # channel_fallback_subjects.csv
    fallback_df.to_csv(out_dir / "channel_fallback_subjects.csv", index=False)
    print(f"  Wrote channel_fallback_subjects.csv ({len(fallback_df)} rows)")

    # training_ready_manifest.csv
    training_manifest.to_csv(out_dir / "training_ready_manifest.csv", index=False)
    print(f"  Wrote training_ready_manifest.csv ({len(training_manifest)} rows)")

    # README.md
    write_readme(
        out_dir,
        tensor_info,
        tensor_ok,
        tensor_issues,
        alignment,
        cohort,
        demo_audit,
        fallback_df,
        ready,
        blocking,
        warn_list,
        _METADATA,
        _MANIFEST,
        run_ts,
    )

    # ---- Summary ----
    print()
    print("=" * 60)
    print(f"READY FOR BASELINE TRAINING: {'YES' if ready else 'NO'}")
    print("=" * 60)
    if blocking:
        print("BLOCKING ISSUES:")
        for b in blocking:
            print(f"  - {b}")
    if warn_list:
        print("WARNINGS:")
        for w in warn_list:
            wrapped = textwrap.fill(w, width=78, subsequent_indent="    ")
            print(f"  - {wrapped}")
    print()
    print(f"Supervised cohort (CN+AD, with demographics): "
          f"{cohort['n_supervised_with_demo']} subjects "
          f"(CN={cohort['supervised_with_demo_group_counts'].get('CN',0)}, "
          f"AD={cohort['supervised_with_demo_group_counts'].get('AD',0)})")
    print(f"VAE cohort (all): {cohort['n_vae_all']} subjects")
    print()
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
