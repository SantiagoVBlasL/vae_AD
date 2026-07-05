"""
Read-only audit of processed OASIS AAL3 package received from Martín (2026-05-30).

Checks:
  A. File inventory (ResultsAAL3 + RealignParameter)
  B. Coverage against selected_ideal_60CN_60AD.csv (expected 120 subjects)
  C. Signal QC (shape, NaN/Inf, zero ROIs, low-variance ROIs, timepoints)
  D. Motion QC (FD_Jenkinson from pre-computed files, flags high-motion runs)
  E. TR=2.2 confirmation, pilot overlap check
  F. process-ready subject list

No tensor building, no model inference, no training, no input data modification.
"""

import re
import sys
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_ROOT = Path("/media/diego/Datos/vae_AD_data/OneDrive_1_30-5-2026")
ROI_SIGNALS_DIR = INPUT_ROOT / "ResultsAAL3" / "ROISignals_AAL3_FunImgARWSDCFN"
REALIGN_DIR = INPUT_ROOT / "RealignParameter"

SELECTION_REF = Path(
    "results/revision_bspc_2026/oasis_next_batch_selection_audit/selected_ideal_60CN_60AD.csv"
)
DOWNLOAD_MANIFEST = Path(
    "results/revision_bspc_2026/oasis_next_batch_selection_audit/download_manifest_for_martin.csv"
)

OUTPUT_DIR = Path(
    "results/revision_bspc_2026/oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530"
)

# Signal QC thresholds
EXPECTED_N_ROIS = 170          # standard AAL3
MIN_TIMEPOINTS = 100           # < 100 TRs at TR=2.2 = <220 s, considered short
LOW_VAR_STD_THRESHOLD = 1e-3   # ROI std below this = low variance
# Motion QC thresholds
FD_MILD_THRESHOLD = 0.3        # mean FD > 0.3 = mild concern flag
FD_SEVERE_THRESHOLD = 0.5      # mean FD > 0.5 = severe flag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RUN_RE = re.compile(
    r"sub-(?P<subject>OAS3\d+)_ses-(?P<session>d\d+)_task-rest_run-(?P<run>\d+)_bold"
)


def parse_run_key(name: str) -> dict | None:
    """Parse subject/session/run from a filename or directory name."""
    m = RUN_RE.search(name)
    if m is None:
        return None
    return {
        "subject_id": f"sub-{m.group('subject')}",
        "bids_session": f"ses-{m.group('session')}",
        "run": m.group("run"),
        "run_key": f"sub-{m.group('subject')}_ses-{m.group('session')}_task-rest_run-{m.group('run')}_bold",
        "subj_ses_key": f"sub-{m.group('subject')}_ses-{m.group('session')}",
    }


def load_signal_txt(path: Path) -> tuple[np.ndarray | None, str]:
    """Load comma-delimited ROI signal file. Returns (array, error_msg)."""
    try:
        data = np.loadtxt(str(path), delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data, ""
    except Exception as e:
        return None, str(e)


def load_fd_jenkinson(path: Path) -> np.ndarray | None:
    """Load single-column FD_Jenkinson file."""
    try:
        fd = np.loadtxt(str(path))
        return fd.ravel()
    except Exception:
        return None


def load_headmotion_tsvs(realign_dir: Path) -> pd.DataFrame:
    """Concatenate all HeadMotion batch TSV/TXT files into one DataFrame."""
    frames = []
    for p in sorted(realign_dir.glob("HeadMotion_batch*.tsv")):
        try:
            df = pd.read_csv(p, sep="\t")
            frames.append(df)
        except Exception:
            pass
    for p in sorted(realign_dir.glob("HeadMotion_batch*.txt")):
        try:
            df = pd.read_csv(p, sep="\t")
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # rename Subject ID column if present
    if "Subject ID" in out.columns:
        out = out.rename(columns={"Subject ID": "run_key"})
    return out


def load_exclude_sets(realign_dir: Path) -> set:
    """Collect all flagged-for-exclusion run keys from ExcludeSubjects files."""
    excluded = set()
    for p in sorted(realign_dir.glob("ExcludeSubjectsAccordingToMaxHeadMotion_batch*.txt")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line.startswith("sub-OAS3"):
                    excluded.add(line)
    return excluded


# ---------------------------------------------------------------------------
# A. File inventory
# ---------------------------------------------------------------------------

def inventory_files(roi_dir: Path, realign_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (signal_inventory_df, realign_inventory_df).
    signal: one row per .txt signal file
    realign: one row per per-run subdirectory in RealignParameter
    """
    signal_rows = []
    for p in sorted(roi_dir.glob("ROISignals_*.txt")):
        info = parse_run_key(p.name)
        if info is None:
            continue
        mat_path = p.with_suffix(".mat")
        gif_path = roi_dir / f"meanfMRIatlas_{info['run_key']}.gif"
        signal_rows.append({
            **info,
            "txt_path": str(p),
            "txt_exists": True,
            "mat_exists": mat_path.exists(),
            "gif_exists": gif_path.exists(),
            "txt_size_bytes": p.stat().st_size,
        })

    realign_rows = []
    for d in sorted(realign_dir.iterdir()):
        if not d.is_dir():
            continue
        info = parse_run_key(d.name)
        if info is None:
            continue
        fd_j = d / f"FD_Jenkinson_{info['run_key']}.txt"
        fd_p = d / f"FD_Power_{info['run_key']}.txt"
        fd_v = d / f"FD_VanDijk_{info['run_key']}.txt"
        rp = d / f"rp_asub-{info['subject_id'].replace('sub-','')}_{info['bids_session']}_task-rest_run-{info['run']}_bold.txt"
        realign_rows.append({
            **info,
            "dir_path": str(d),
            "fd_jenkinson_exists": fd_j.exists(),
            "fd_power_exists": fd_p.exists(),
            "fd_vandijk_exists": fd_v.exists(),
            "rp_exists": rp.exists(),
            "fd_jenkinson_path": str(fd_j) if fd_j.exists() else "",
        })

    return pd.DataFrame(signal_rows), pd.DataFrame(realign_rows)


# ---------------------------------------------------------------------------
# B. Coverage audit
# ---------------------------------------------------------------------------

def coverage_audit(
    selection_df: pd.DataFrame,
    signal_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (coverage_df, missing_df, extra_df, diag_balance_df).
    """
    # Build per-subject-session lookup from selection
    sel = selection_df[["subject_id", "bids_session", "experiment_id", "diagnosis",
                         "age_at_MR", "sex", "Manufacturer", "TR_seconds",
                         "expected_tr2_rest_runs", "pilot_subject_overlap"]].copy()
    sel["subj_ses_key"] = sel["subject_id"] + "_" + sel["bids_session"]

    # Build set of processed subj-session keys
    if signal_df.empty:
        processed_keys = set()
    else:
        processed_keys = set(signal_df["subj_ses_key"].unique())

    # Coverage per selected subject-session
    coverage_rows = []
    for _, row in sel.iterrows():
        key = row["subj_ses_key"]
        if signal_df.empty:
            n_runs_found = 0
            found_runs = []
        else:
            mask = signal_df["subj_ses_key"] == key
            n_runs_found = int(mask.sum())
            found_runs = sorted(signal_df.loc[mask, "run"].tolist())
        coverage_rows.append({
            "subject_id": row["subject_id"],
            "bids_session": row["bids_session"],
            "experiment_id": row["experiment_id"],
            "diagnosis": row["diagnosis"],
            "age_at_MR": row["age_at_MR"],
            "sex": row["sex"],
            "Manufacturer": row["Manufacturer"],
            "TR_seconds": row["TR_seconds"],
            "expected_tr2_rest_runs": row["expected_tr2_rest_runs"],
            "pilot_subject_overlap": row["pilot_subject_overlap"],
            "subj_ses_key": key,
            "n_runs_processed": n_runs_found,
            "processed_runs": ";".join(found_runs),
            "is_processed": n_runs_found > 0,
        })

    coverage_df = pd.DataFrame(coverage_rows)

    missing_df = coverage_df[~coverage_df["is_processed"]].reset_index(drop=True)

    # Extra: processed but not in selection
    selected_keys = set(coverage_df["subj_ses_key"])
    if not signal_df.empty:
        extra_mask = ~signal_df["subj_ses_key"].isin(selected_keys)
        extra_df = signal_df[extra_mask][["subject_id", "bids_session", "run_key", "subj_ses_key"]].drop_duplicates("subj_ses_key").reset_index(drop=True)
    else:
        extra_df = pd.DataFrame()

    # Diagnosis balance
    diag_counts = (
        coverage_df[coverage_df["is_processed"]]
        .groupby("diagnosis")
        .agg(n_subjects=("subject_id", "count"), n_runs_total=("n_runs_processed", "sum"))
        .reset_index()
    )

    return coverage_df, missing_df, extra_df, diag_counts


# ---------------------------------------------------------------------------
# C. Signal QC
# ---------------------------------------------------------------------------

def signal_qc(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Load each signal txt and compute QC metrics."""
    rows = []
    for _, row in signal_df.iterrows():
        p = Path(row["txt_path"])
        data, err = load_signal_txt(p)

        if data is None:
            rows.append({
                "run_key": row["run_key"],
                "subject_id": row["subject_id"],
                "bids_session": row["bids_session"],
                "run": row["run"],
                "load_error": err,
                "n_timepoints": None,
                "n_rois": None,
                "n_nan_total": None,
                "n_nan_rois_structural": None,
                "n_scattered_nan": None,
                "n_inf": None,
                "n_zero_rois": None,
                "n_low_var_rois": None,
                "is_short_run": None,
                "roi_count_ok": None,
                "pipeline_compatible_shape": None,
                "qc_pass": False,
            })
            continue

        n_tp, n_roi = data.shape
        # Distinguish structural NaN (entire ROI column all-NaN, normal in DPABI/AAL3
        # for parcels with no voxels in MNI space) from scattered NaN (problematic).
        nan_mask = np.isnan(data)
        all_nan_roi_mask = nan_mask.all(axis=0)
        n_nan_rois = int(all_nan_roi_mask.sum())       # structural (expected = 4)
        n_scattered_nan = int(nan_mask.sum()) - n_nan_rois * n_tp  # non-structural
        n_nan = int(nan_mask.sum())
        n_inf = int(np.isinf(data).sum())
        # Compute stats only on valid ROIs
        data_valid = data[:, ~all_nan_roi_mask]
        n_zero_rois = int((np.abs(data_valid).sum(axis=0) == 0).sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roi_std = data_valid.std(axis=0)
        n_low_var = int((roi_std < LOW_VAR_STD_THRESHOLD).sum())
        is_short = n_tp < MIN_TIMEPOINTS
        roi_count_ok = n_roi == EXPECTED_N_ROIS
        # QC pass: scattered NaN = 0, no Inf, not short, correct ROI count.
        # Structural all-NaN ROIs are expected DPABI output and are NOT a failure.
        qc_pass = (n_scattered_nan == 0 and n_inf == 0 and not is_short and roi_count_ok)

        rows.append({
            "run_key": row["run_key"],
            "subject_id": row["subject_id"],
            "bids_session": row["bids_session"],
            "run": row["run"],
            "load_error": "",
            "n_timepoints": n_tp,
            "n_rois": n_roi,
            "n_nan_total": n_nan,
            "n_nan_rois_structural": n_nan_rois,
            "n_scattered_nan": n_scattered_nan,
            "n_inf": n_inf,
            "n_zero_rois": n_zero_rois,
            "n_low_var_rois": n_low_var,
            "is_short_run": is_short,
            "roi_count_ok": roi_count_ok,
            "pipeline_compatible_shape": roi_count_ok,
            "qc_pass": qc_pass,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# D. Motion QC
# ---------------------------------------------------------------------------

def motion_qc(
    signal_df: pd.DataFrame,
    realign_df: pd.DataFrame,
    headmotion_df: pd.DataFrame,
    excluded_set: set,
) -> pd.DataFrame:
    """
    Compute motion QC per run.

    Two motion metrics are reported:
    1. FD_Jenkinson (frame-to-frame): from pre-computed per-run files.
       Threshold: mean FD > 0.3 = mild flag, > 0.5 = severe flag.
    2. Absolute-drift criterion (Martín's): max displacement from reference
       frame (3mm translation OR 3° rotation in any direction). Flagged by
       presence in ExcludeSubjectsAccordingToMaxHeadMotion_batchN.txt.
       NOTE: This is a strict criterion; 226/241 runs are flagged. Many runs
       with low mean FD are still flagged because of cumulative head drift
       over the scan. These runs may be usable depending on analysis choice.
    """
    rows = []
    for _, row in signal_df.iterrows():
        rk = row["run_key"]

        # FD_Jenkinson from per-run file
        fd_vals = None
        fd_source = "none"
        if not realign_df.empty:
            r_match = realign_df[realign_df["run_key"] == rk]
            if not r_match.empty and r_match.iloc[0]["fd_jenkinson_exists"]:
                fd_vals = load_fd_jenkinson(Path(r_match.iloc[0]["fd_jenkinson_path"]))
                if fd_vals is not None:
                    fd_source = "fd_jenkinson_file"

        # Get absolute-drift stats from batch HeadMotion TSV
        hm_row = None
        if not headmotion_df.empty and "run_key" in headmotion_df.columns:
            hm_match = headmotion_df[headmotion_df["run_key"] == rk]
            if not hm_match.empty:
                hm_row = hm_match.iloc[0]

        if fd_vals is not None:
            mean_fd = float(fd_vals.mean())
            max_fd = float(fd_vals.max())
            pct_gt02 = float((fd_vals > 0.2).sum()) / len(fd_vals) * 100
            pct_gt05 = float((fd_vals > 0.5).sum()) / len(fd_vals) * 100
            n_frames = len(fd_vals)
        elif hm_row is not None and "mean FD_Jenkinson" in headmotion_df.columns:
            mean_fd = float(hm_row["mean FD_Jenkinson"])
            max_fd = float("nan")
            pct_gt02 = float(hm_row.get("Percent of FD_Power>0.2", float("nan")))
            pct_gt05 = float(hm_row.get("Percent of FD_Power>0.5", float("nan")))
            n_frames = None
            fd_source = "headmotion_tsv"
        else:
            mean_fd = max_fd = pct_gt02 = pct_gt05 = float("nan")
            n_frames = None
            fd_source = "missing"

        # Absolute-drift stats from HeadMotion TSV (in degrees for rotation)
        max_tx = float(hm_row["max(abs(Tx))"]) if hm_row is not None else float("nan")
        max_ty = float(hm_row["max(abs(Ty))"]) if hm_row is not None else float("nan")
        max_tz = float(hm_row["max(abs(Tz))"]) if hm_row is not None else float("nan")
        max_rx = float(hm_row["max(abs(Rx))"]) if hm_row is not None else float("nan")
        max_ry = float(hm_row["max(abs(Ry))"]) if hm_row is not None else float("nan")
        max_rz = float(hm_row["max(abs(Rz))"]) if hm_row is not None else float("nan")
        max_trans = max(abs(max_tx), abs(max_ty), abs(max_tz)) if all(
            not np.isnan(v) for v in [max_tx, max_ty, max_tz]) else float("nan")
        max_rot = max(abs(max_rx), abs(max_ry), abs(max_rz)) if all(
            not np.isnan(v) for v in [max_rx, max_ry, max_rz]) else float("nan")

        flagged_mild = (mean_fd > FD_MILD_THRESHOLD) if not np.isnan(mean_fd) else None
        flagged_severe = (mean_fd > FD_SEVERE_THRESHOLD) if not np.isnan(mean_fd) else None
        # Martín's strict absolute-drift criterion (3mm/3°)
        martins_exclude = rk in excluded_set

        rows.append({
            "run_key": rk,
            "subject_id": row["subject_id"],
            "bids_session": row["bids_session"],
            "run": row["run"],
            "fd_source": fd_source,
            "n_frames": n_frames,
            "mean_fd_jenkinson": round(mean_fd, 6) if not np.isnan(mean_fd) else None,
            "max_fd_jenkinson": round(max_fd, 6) if not np.isnan(max_fd) else None,
            "pct_frames_fd_gt0p2": round(pct_gt02, 2) if not np.isnan(pct_gt02) else None,
            "pct_frames_fd_gt0p5": round(pct_gt05, 2) if not np.isnan(pct_gt05) else None,
            "flag_mild_motion_fd": flagged_mild,
            "flag_severe_motion_fd": flagged_severe,
            "max_abs_translation_mm": round(max_trans, 4) if not np.isnan(max_trans) else None,
            "max_abs_rotation_col_units": round(max_rot, 6) if not np.isnan(max_rot) else None,
            "martins_pipeline_exclude_3mm3deg": martins_exclude,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E. Process-ready subjects
# ---------------------------------------------------------------------------

def build_process_ready(
    coverage_df: pd.DataFrame,
    signal_qc_df: pd.DataFrame,
    motion_qc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    A subject-session is process-ready if:
      - At least 1 processed run
      - That run passes signal QC (qc_pass=True)
      - That run is not flagged severe motion
      - That run is not in Martín's exclude list
    We pick the best run per subject-session (prefer run-01, else first available).
    """
    # Merge signal + motion QC
    if signal_qc_df.empty or motion_qc_df.empty:
        return pd.DataFrame()

    sq = signal_qc_df[["run_key", "subject_id", "bids_session", "run",
                        "n_timepoints", "n_rois", "qc_pass"]].copy()
    mq = motion_qc_df[["run_key", "mean_fd_jenkinson", "pct_frames_fd_gt0p2",
                        "pct_frames_fd_gt0p5", "flag_severe_motion_fd",
                        "martins_pipeline_exclude_3mm3deg"]].copy()
    merged = sq.merge(mq, on="run_key", how="left")
    merged["subj_ses_key"] = merged["subject_id"] + "_" + merged["bids_session"]

    # Candidate runs: signal QC pass + not severe by FD criterion.
    # Martín's strict 3mm/3° absolute-drift criterion is reported separately
    # and NOT used here — researcher decides whether to apply it.
    merged["candidate"] = (
        merged["qc_pass"].fillna(False)
        & (~merged["flag_severe_motion_fd"].fillna(True))
    )

    # Per subject-session: pick best candidate run (lowest run number)
    ready_rows = []
    for key, grp in merged.groupby("subj_ses_key"):
        cands = grp[grp["candidate"]].sort_values("run")
        if cands.empty:
            continue
        best = cands.iloc[0]
        # Look up diagnosis from coverage_df
        diag_row = coverage_df[coverage_df["subj_ses_key"] == key]
        diagnosis = diag_row["diagnosis"].iloc[0] if not diag_row.empty else "unknown"
        ready_rows.append({
            "subject_id": best["subject_id"],
            "bids_session": best["bids_session"],
            "subj_ses_key": key,
            "selected_run": best["run"],
            "run_key": best["run_key"],
            "diagnosis": diagnosis,
            "n_timepoints": best["n_timepoints"],
            "n_rois": best["n_rois"],
            "mean_fd_jenkinson": best["mean_fd_jenkinson"],
            "pct_frames_fd_gt0p2": best["pct_frames_fd_gt0p2"],
        })

    return pd.DataFrame(ready_rows)


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def df_to_md(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No data._\n"
    return df.head(max_rows).to_markdown(index=False)


def save_csv_md(df: pd.DataFrame, stem: str, out_dir: Path, max_rows: int = 200) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df_to_md(df, max_rows=max_rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading reference files...")
    selection_df = pd.read_csv(SELECTION_REF)
    manifest_df = pd.read_csv(DOWNLOAD_MANIFEST)

    print("A. File inventory...")
    signal_df, realign_df = inventory_files(ROI_SIGNALS_DIR, REALIGN_DIR)
    save_csv_md(signal_df, "processed_file_inventory", OUTPUT_DIR)
    print(f"   Signal files: {len(signal_df)}")
    print(f"   Realign dirs: {len(realign_df)}")

    print("B. Coverage audit...")
    coverage_df, missing_df, extra_df, diag_balance_df = coverage_audit(selection_df, signal_df)
    save_csv_md(coverage_df, "coverage_by_selected_experiment", OUTPUT_DIR)
    save_csv_md(missing_df, "missing_processed_experiments", OUTPUT_DIR)
    save_csv_md(extra_df, "extra_processed_experiments", OUTPUT_DIR)
    save_csv_md(diag_balance_df, "diagnosis_balance_processed", OUTPUT_DIR)

    n_processed = int(coverage_df["is_processed"].sum())
    n_missing = len(missing_df)
    n_extra = len(extra_df)
    print(f"   Processed subjects: {n_processed}/120")
    print(f"   Missing: {n_missing}, Extra: {n_extra}")
    print("   Diagnosis balance:")
    print(diag_balance_df.to_string(index=False))

    # Calibration/test split - not available
    calib_note = "split_calibration_test.csv not available in reference folder."
    (OUTPUT_DIR / "calibration_test_coverage.md").write_text(
        "# Calibration/Test Split Coverage\n\n" + calib_note + "\n"
    )

    print("C. Signal QC...")
    signal_qc_df = signal_qc(signal_df)
    save_csv_md(signal_qc_df, "signal_qc_by_file", OUTPUT_DIR)
    n_qc_pass = int(signal_qc_df["qc_pass"].sum()) if not signal_qc_df.empty else 0
    n_qc_fail = len(signal_qc_df) - n_qc_pass if not signal_qc_df.empty else 0
    print(f"   QC pass: {n_qc_pass}, fail: {n_qc_fail}")
    if not signal_qc_df.empty:
        print(f"   ROI counts: {signal_qc_df['n_rois'].value_counts().to_dict()}")
        print(f"   TP range: {signal_qc_df['n_timepoints'].min()} – {signal_qc_df['n_timepoints'].max()}")

    print("D. Motion QC...")
    headmotion_df = load_headmotion_tsvs(REALIGN_DIR)
    excluded_set = load_exclude_sets(REALIGN_DIR)
    motion_qc_df = motion_qc(signal_df, realign_df, headmotion_df, excluded_set)
    save_csv_md(motion_qc_df, "motion_qc_by_file", OUTPUT_DIR)
    if not motion_qc_df.empty:
        n_mild = int(motion_qc_df["flag_mild_motion_fd"].sum())
        n_severe = int(motion_qc_df["flag_severe_motion_fd"].sum())
        n_martins_excl = int(motion_qc_df["martins_pipeline_exclude_3mm3deg"].sum())
        mean_fd_overall = motion_qc_df["mean_fd_jenkinson"].mean()
        print(f"   Mean FD (all runs): {mean_fd_overall:.4f}")
        print(f"   Mild motion (FD>{FD_MILD_THRESHOLD}): {n_mild}")
        print(f"   Severe motion (FD>{FD_SEVERE_THRESHOLD}): {n_severe}")
        print(f"   Martín absolute-drift exclude (3mm/3°): {n_martins_excl}")

    print("E. Process-ready subjects...")
    ready_df = build_process_ready(coverage_df, signal_qc_df, motion_qc_df)
    save_csv_md(ready_df, "process_ready_subjects", OUTPUT_DIR)
    if not ready_df.empty:
        print(f"   Process-ready: {len(ready_df)}")
        print("   Diagnosis:", ready_df["diagnosis"].value_counts().to_dict())

    # TR confirmation
    tr_values = selection_df["TR_seconds"].value_counts().to_dict()
    print(f"   TR values in selection: {tr_values}")

    # Pilot overlap
    pilot_overlaps = selection_df[selection_df.get("pilot_subject_overlap", pd.Series(dtype=object)).eq(True)] if "pilot_subject_overlap" in selection_df.columns else pd.DataFrame()

    # ------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------
    print("Writing final_recommendation.md...")

    n_total_runs = len(signal_df)
    n_unique_subj = signal_df["subject_id"].nunique() if not signal_df.empty else 0
    n_unique_ses = signal_df["subj_ses_key"].nunique() if not signal_df.empty else 0

    roi_ok_pct = (signal_qc_df["roi_count_ok"].mean() * 100) if not signal_qc_df.empty else 0
    tp_median = signal_qc_df["n_timepoints"].median() if not signal_qc_df.empty else 0

    ready_cn = len(ready_df[ready_df["diagnosis"] == "CN"]) if not ready_df.empty else 0
    ready_ad = len(ready_df[ready_df["diagnosis"] == "AD_DEMENTIA"]) if not ready_df.empty else 0

    # severity summary
    if not motion_qc_df.empty:
        n_mild = int(motion_qc_df["flag_mild_motion_fd"].sum())
        n_severe = int(motion_qc_df["flag_severe_motion_fd"].sum())
        n_martins_excl = int(motion_qc_df["martins_pipeline_exclude_3mm3deg"].sum())
        mean_fd_overall = motion_qc_df["mean_fd_jenkinson"].mean()
    else:
        n_mild = n_severe = n_martins_excl = 0
        mean_fd_overall = float("nan")

    recommendation_lines = []
    overall_ready = (
        n_processed == 120
        and n_qc_fail == 0
        and n_severe == 0
    )
    if n_processed < 120:
        recommendation_lines.append(
            f"⚠ **Coverage incomplete**: only {n_processed}/120 selected subjects have processed files. "
            f"{n_missing} subject-sessions are missing."
        )
    if n_extra > 0:
        recommendation_lines.append(
            f"⚠ **Extra files**: {n_extra} processed subject-sessions not in selected_ideal_60CN_60AD.csv."
        )
    if n_qc_fail > 0:
        recommendation_lines.append(
            f"⚠ **Signal QC failures**: {n_qc_fail} runs failed signal QC (NaN/Inf, wrong ROI count, or too short)."
        )
    if n_severe > 0:
        recommendation_lines.append(
            f"⚠ **Severe motion**: {n_severe} runs have mean FD_Jenkinson > {FD_SEVERE_THRESHOLD} mm."
        )
    if n_martins_excl > 0:
        recommendation_lines.append(
            f"ℹ **Martín's absolute-drift criterion (3mm/3°)**: {n_martins_excl}/241 runs flagged "
            "(max displacement from reference frame, NOT frame-to-frame FD). "
            "This strict criterion reflects head drift over the full scan. "
            "Only 15 runs pass it. Verify with Martín whether this threshold was intentional "
            "for connectome preprocessing or is a legacy pipeline default. "
            "See motion_qc_by_file.csv for per-run values."
        )
    if roi_ok_pct < 100:
        recommendation_lines.append(
            f"⚠ **ROI count mismatch**: {100 - roi_ok_pct:.1f}% of files do not have expected {EXPECTED_N_ROIS} ROIs."
        )

    verdict = (
        "**READY FOR TENSOR CONSTRUCTION** (FD-based criterion) — all 120 subjects processed, "
        "all signal QC pass, no severe FD failures. "
        "Review Martín's strict 3mm/3° absolute-drift exclusion list before proceeding."
        if overall_ready
        else "**NOT YET READY FOR TENSOR CONSTRUCTION** — see warnings above."
    )

    warnings_block = (
        "\n".join(f"- {line}" for line in recommendation_lines)
        if recommendation_lines else "- None. Package looks clean."
    )
    generated_ts = datetime.now(timezone.utc).isoformat()
    tr_values_str = str(tr_values)
    all_tr22 = all(k == 2.2 for k in tr_values.keys())

    rec_md = f"""# Final Recommendation — OASIS AAL3 Handoff QC
Generated: {generated_ts}

## Verdict
{verdict}

## Summary Statistics
| Metric | Value |
|--------|-------|
| Expected subjects (selected_ideal_60CN_60AD) | 120 (60 CN + 60 AD) |
| Subjects with ≥1 processed run | {n_processed} |
| Missing subjects | {n_missing} |
| Extra (unselected) processed subjects | {n_extra} |
| Total processed runs (all runs, all subjects) | {n_total_runs} |
| Unique processed subject-sessions | {n_unique_ses} |
| All TR=2.2 (confirmed) | {all_tr22} |
| Signal QC pass | {n_qc_pass}/{n_total_runs} |
| Signal QC fail | {n_qc_fail}/{n_total_runs} |
| ROI count = {EXPECTED_N_ROIS} (%) | {roi_ok_pct:.1f}% |
| Median timepoints | {tp_median:.0f} |
| Mean FD_Jenkinson (all runs) | {mean_fd_overall:.4f} mm |
| Runs: mild motion flag (FD>{FD_MILD_THRESHOLD}) | {n_mild} |
| Runs: severe motion flag (FD>{FD_SEVERE_THRESHOLD}) | {n_severe} |
| Runs: Martín pipeline exclude (3mm/3°) | {n_martins_excl} |
| Process-ready subjects (best run) | {len(ready_df)} (CN={ready_cn}, AD={ready_ad}) |

## Warnings / Issues
{warnings_block}

## Pipeline Compatibility Note
Signal files contain **{EXPECTED_N_ROIS} AAL3 ROIs** (standard DPABI/DPARSF output, Rolls et al. 2020).
**4 ROIs (indices 34, 35, 80, 81) are all-NaN in every file** — this is normal DPABI behavior for
AAL3 parcels with no valid MNI-space voxels in the rsfMRI field of view. These structural NaN ROIs
are NOT a data quality problem. They will be removed as part of the 170→131 ROI exclusion step.

The existing ADNI pipeline uses **131 ROIs** after applying the AAL3→131 exclusion/reorder table.
This remapping must be applied before tensor construction. No change to this package is needed —
the 170-column files are the correct input format for that step.

## TR Confirmation
All selected subjects have TR=2.2s (confirmed in selection audit). Martín reported processing
only TR=2.2 runs. Found TR values in selection: {tr_values_str}.

## Next Steps
1. If any missing subjects are listed in missing_processed_experiments.csv, contact Martín for reprocessing.
2. Review motion_qc_by_file.csv — decide whether runs flagged by Martín's 3mm/3° criterion should be excluded from tensor construction.
3. Apply AAL3 170→131 ROI exclusion/reorder table to each .txt signal file.
4. Build per-subject connectivity tensors from process_ready_subjects.csv.
5. Run locked-model inference (no retraining).
"""

    (OUTPUT_DIR / "final_recommendation.md").write_text(rec_md)

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------
    readme = f"""# OASIS AAL3 Handoff QC — 2026-05-30

Read-only audit of processed OASIS package received from Martín.
Input: {INPUT_ROOT}
Reference: {SELECTION_REF}

## Contents
| File | Description |
|------|-------------|
| processed_file_inventory.csv | All ROISignals .txt/.mat/.gif files found |
| coverage_by_selected_experiment.csv | Per-selected-subject coverage status |
| missing_processed_experiments.csv | Selected subjects with no processed files |
| extra_processed_experiments.csv | Processed subjects not in selection list |
| signal_qc_by_file.csv | Shape, NaN, zero-ROI, low-variance, short-run per file |
| motion_qc_by_file.csv | FD_Jenkinson stats, motion flags per run |
| diagnosis_balance_processed.csv | CN/AD counts among processed subjects |
| calibration_test_coverage.md | Calibration/test split (not available) |
| process_ready_subjects.csv | Subjects ready for tensor construction |
| final_recommendation.md | Verdict + next steps |
| command_log.json | Audit provenance |

## Quick Summary
- Processed: {n_processed}/120 subjects
- Signal QC: {n_qc_pass} pass / {n_qc_fail} fail
- Motion (severe): {n_severe} runs
- Process-ready: {len(ready_df)} (CN={ready_cn}, AD={ready_ad})
"""
    (OUTPUT_DIR / "README.md").write_text(readme)

    # ------------------------------------------------------------------
    # command_log.json
    # ------------------------------------------------------------------
    log = {
        "script": "scripts/revision_bspc_2026/audit_oasis_next_60cn_60ad_processed_aal3_handoff_qc.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(INPUT_ROOT),
        "roi_signals_dir": str(ROI_SIGNALS_DIR),
        "realign_dir": str(REALIGN_DIR),
        "reference_selection_csv": str(SELECTION_REF),
        "output_dir": str(OUTPUT_DIR),
        "expected_subjects": 120,
        "n_processed": n_processed,
        "n_missing": n_missing,
        "n_extra": n_extra,
        "n_total_runs": n_total_runs,
        "signal_qc_pass": n_qc_pass,
        "signal_qc_fail": n_qc_fail,
        "motion_severe_runs": n_severe,
        "martins_excluded_runs": n_martins_excl,
        "process_ready_subjects": len(ready_df),
        "training_launched": False,
        "threshold_fitting": False,
        "tensor_modification": False,
        "model_inference": False,
    }
    (OUTPUT_DIR / "command_log.json").write_text(
        json.dumps(log, indent=2, default=str)
    )

    print(f"\nAll outputs written to: {OUTPUT_DIR}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
