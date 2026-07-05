#!/usr/bin/env python3
"""Build OASIS 60 CN / 60 AD connectome tensors from processed AAL3 signals.

Three build candidates are computed for every selected subject:

  concatenated_timeseries
      All QC-passing runs for the subject are concatenated along the time axis.
      A single connectome is computed from the concatenated series.

  runwise_140TR_connectome_average  (ADNI-like sensitivity)
      For each run, only the first 140 timepoints are used.
      Raw channel matrices are computed per run, then averaged, then normalized.

  runwise164_connectome_average  (supplementary)
      All 164 timepoints are used per run.
      Same average-then-normalize scheme as above.

Channel order in tensor (matches tanda build; inference selects by name):
  0  Pearson_Full_FisherZ_Signed
  1  Pearson_OMST_GCE_Signed_Weighted
  2  MI_KNN_Symmetric

ROI mapping: reuses the confirmed tanda preflight mapping (131 ROIs in ADNI
final order, oasis_col_idx_0based derived from AAL3 color indices).

Safety:
  - Default run is read-only / dry-run.
  - Pass --confirm-build to actually compute connectomes.
  - No tensor building, no model inference, no training, no input modification.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_ROOT = Path("/media/diego/Datos/vae_AD_data/OneDrive_1_30-5-2026")
ROI_SIGNALS_DIR = INPUT_ROOT / "ResultsAAL3" / "ROISignals_AAL3_FunImgARWSDCFN"

HANDOFF_QC_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530"
)
SELECTION_AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_next_batch_selection_audit"
)
TANDA_PREFLIGHT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectome_build_preflight"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_next_60cn_60ad_tensor_build_20260530"
)

CHANNEL_NAMES: list[str] = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
BUILD_CANDIDATES: list[str] = [
    "concatenated_timeseries",
    "runwise_140TR_connectome_average",
    "runwise164_connectome_average",
]
N_NEIGHBORS_MI: int = 5
TR_LIMIT_140: int = 140
EXPECTED_N_ROIS: int = 131
EXPECTED_N_ROIS_RAW: int = 170


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--confirm-build", action="store_true",
        help="Required to actually compute connectomes and write tensor NPZ files.",
    )
    parser.add_argument(
        "--build-candidate",
        choices=["all"] + BUILD_CANDIDATES,
        default="all",
        help="Which build candidate(s) to run (default: all three).",
    )
    parser.add_argument(
        "--n-jobs-mi", type=int, default=8,
        help="Parallel jobs for MI_KNN pair computation (default: 8).",
    )
    parser.add_argument(
        "--fail-if-output-exists", action="store_true",
        help="Refuse to overwrite an existing tensor NPZ.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


def _selected_candidates(build_candidate: str) -> list[str]:
    return BUILD_CANDIDATES if build_candidate == "all" else [build_candidate]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate all input tables.

    Returns
    -------
    process_ready, signal_qc, motion_qc, selected_ideal, roi_mapping
    """
    process_ready = pd.read_csv(HANDOFF_QC_DIR / "process_ready_subjects.csv")
    signal_qc = pd.read_csv(HANDOFF_QC_DIR / "signal_qc_by_file.csv")
    motion_qc = pd.read_csv(HANDOFF_QC_DIR / "motion_qc_by_file.csv")
    selected_ideal = pd.read_csv(SELECTION_AUDIT_DIR / "selected_ideal_60CN_60AD.csv")
    roi_mapping = pd.read_csv(TANDA_PREFLIGHT_DIR / "roi_mapping_used.csv")

    n_subj = len(process_ready)
    if n_subj != 120:
        raise ValueError(f"Expected 120 process-ready subjects, got {n_subj}")
    if len(roi_mapping) != EXPECTED_N_ROIS:
        raise ValueError(f"Expected {EXPECTED_N_ROIS} ROI mapping rows, got {len(roi_mapping)}")
    if not roi_mapping["matches_adni_final_order"].astype(bool).all():
        raise ValueError("ROI mapping 'matches_adni_final_order' is not all True — refusing to build")

    return process_ready, signal_qc, motion_qc, selected_ideal, roi_mapping


def build_run_table(
    process_ready: pd.DataFrame,
    signal_qc: pd.DataFrame,
    motion_qc: pd.DataFrame,
) -> pd.DataFrame:
    """Build the run inclusion table.

    Includes all runs that:
    - belong to a process-ready subject
    - pass signal QC (qc_pass == True)
    - do not have severe FD motion (flag_severe_motion_fd == False)
    """
    selected_subjects = set(process_ready["subject_id"])

    sig = signal_qc[signal_qc["qc_pass"].astype(bool)].copy()
    mot_cols = motion_qc[["run_key", "flag_severe_motion_fd", "mean_fd_jenkinson"]].copy()
    merged = sig.merge(mot_cols, on="run_key", how="left")

    merged = merged[~merged["flag_severe_motion_fd"].fillna(False)]
    merged = merged[merged["subject_id"].isin(selected_subjects)].copy()

    merged["roi_txt_path"] = merged["run_key"].apply(
        lambda k: str(ROI_SIGNALS_DIR / f"ROISignals_{k}.txt")
    )

    return merged.sort_values(["subject_id", "bids_session", "run"]).reset_index(drop=True)


def build_subject_manifest(
    process_ready: pd.DataFrame,
    selected_ideal: pd.DataFrame,
    run_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a subject-level manifest joining process-ready list with selection metadata."""
    meta = selected_ideal.drop_duplicates("subject_id")[
        ["subject_id", "session_id", "experiment_id",
         "age_at_MR", "sex", "Manufacturer", "ScannerModel"]
    ].copy()

    manifest = process_ready[["subject_id", "bids_session", "diagnosis"]].merge(
        meta, on="subject_id", how="left"
    )

    runs_agg = (
        run_table.groupby("subject_id")
        .agg(
            selected_qc_runs=("run", "count"),
            selected_run_ids=("run", lambda x: ";".join(sorted(str(r) for r in x))),
            selected_total_timepoints=("n_timepoints", "sum"),
        )
        .reset_index()
    )
    manifest = manifest.merge(runs_agg, on="subject_id", how="left")

    return manifest.sort_values("subject_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Channel computation helpers
# ---------------------------------------------------------------------------

def fisher_r_to_z(r_matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r_clean = np.nan_to_num(r_matrix.astype(np.float32), nan=0.0)
    r_clipped = np.clip(r_clean, -1.0 + eps, 1.0 - eps)
    z = np.arctanh(r_clipped)
    np.fill_diagonal(z, 0.0)
    return z.astype(np.float32)


def pearson_full(ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ts, rowvar=False).astype(np.float32)
    return fisher_r_to_z(corr)


def pearson_omst(ts: np.ndarray) -> np.ndarray:
    try:
        from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
    except Exception as exc:
        raise RuntimeError("dyconnmap OMST dependency unavailable") from exc

    z = pearson_full(ts)
    weights = np.abs(z)
    np.fill_diagonal(weights, 0.0)
    if np.all(np.isclose(weights, 0)):
        return z
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
        warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)
        outputs = threshold_omst_global_cost_efficiency(weights, n_msts=None)
    if not (isinstance(outputs, tuple) and len(outputs) >= 2):
        raise RuntimeError(f"Unexpected OMST return: {type(outputs)}")
    omst_adj = np.asarray(outputs[1]).astype(np.float32)
    mask = (omst_adj > 0).astype(np.float32)
    out = z * mask
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def _mi_pair(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> float:
    try:
        return float(
            mutual_info_regression(
                x.reshape(-1, 1), y,
                n_neighbors=n_neighbors, random_state=42, discrete_features=False,
            )[0]
        )
    except Exception:
        return 0.0


def mi_knn_symmetric(
    ts: np.ndarray, n_neighbors: int = N_NEIGHBORS_MI, n_jobs: int = 1,
) -> np.ndarray:
    from joblib import Parallel, delayed

    n_tp, n_rois = ts.shape
    if n_tp <= n_neighbors:
        return np.zeros((n_rois, n_rois), dtype=np.float32)

    pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]
    if n_jobs == 1:
        vals = [
            (_mi_pair(ts[:, i], ts[:, j], n_neighbors),
             _mi_pair(ts[:, j], ts[:, i], n_neighbors))
            for i, j in pairs
        ]
    else:
        vals_ij = Parallel(n_jobs=n_jobs)(
            delayed(_mi_pair)(ts[:, i], ts[:, j], n_neighbors) for i, j in pairs
        )
        vals_ji = Parallel(n_jobs=n_jobs)(
            delayed(_mi_pair)(ts[:, j], ts[:, i], n_neighbors) for i, j in pairs
        )
        vals = list(zip(vals_ij, vals_ji))

    mat = np.zeros((n_rois, n_rois), dtype=np.float32)
    for (i, j), (v_ij, v_ji) in zip(pairs, vals):
        mat[i, j] = mat[j, i] = (v_ij + v_ji) / 2.0
    return mat


def normalize_channel_offdiag(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float32)
    mask = ~np.eye(n, dtype=bool)
    vals = matrix[mask]
    if vals.size == 0 or np.nanstd(vals) < 1e-12:
        out[mask] = vals.astype(np.float32)
        return out
    scaled = RobustScaler().fit_transform(vals.reshape(-1, 1)).ravel()
    out[mask] = scaled.astype(np.float32)
    return out


def compute_raw_channels(ts: np.ndarray, n_jobs_mi: int) -> np.ndarray:
    """Return (3, n_rois, n_rois) unnormalized channel stack."""
    return np.stack(
        [pearson_full(ts), pearson_omst(ts), mi_knn_symmetric(ts, n_jobs=n_jobs_mi)],
        axis=0,
    ).astype(np.float32)


def normalize_channels(raw: np.ndarray) -> np.ndarray:
    """Per-subject, per-channel off-diagonal RobustScaler normalization."""
    return np.stack(
        [normalize_channel_offdiag(raw[c]) for c in range(raw.shape[0])],
        axis=0,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# ROI timeseries loading
# ---------------------------------------------------------------------------

def load_roi_timeseries(txt_path: str, roi_mapping: pd.DataFrame, sid: str) -> np.ndarray:
    """Load a 170-column ROI signal file and project to 131 ADNI-ordered ROIs."""
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing ROI timeseries for {sid}: {path}")

    matrix = np.loadtxt(path, delimiter=",")
    if matrix.ndim != 2:
        raise ValueError(f"{sid}: ROI matrix is not 2D: shape={matrix.shape}")

    if matrix.shape[1] == EXPECTED_N_ROIS_RAW:
        ts_170 = matrix
    elif matrix.shape[0] == EXPECTED_N_ROIS_RAW:
        ts_170 = matrix.T
    else:
        raise ValueError(
            f"{sid}: expected one dimension == {EXPECTED_N_ROIS_RAW}, got {matrix.shape}"
        )

    cols = roi_mapping["oasis_col_idx_0based"].astype(int).to_numpy()
    ts_131 = ts_170[:, cols].astype(np.float32)

    if ts_131.shape[1] != EXPECTED_N_ROIS:
        raise ValueError(f"{sid}: post-mapping ROI count mismatch: {ts_131.shape}")

    all_nan_cols = np.where(np.isnan(ts_131).all(axis=0))[0].tolist()
    if all_nan_cols:
        raise ValueError(f"{sid}: retained ADNI ROI cols are all-NaN after mapping: {all_nan_cols}")

    if not np.isfinite(ts_131).all():
        median_val = float(np.nanmedian(ts_131))
        ts_131 = np.nan_to_num(ts_131, nan=median_val, posinf=0.0, neginf=0.0)

    return ts_131


# ---------------------------------------------------------------------------
# Subject tensor computation
# ---------------------------------------------------------------------------

def build_subject_tensor(
    candidate: str,
    run_rows: pd.DataFrame,
    roi_mapping: pd.DataFrame,
    n_jobs_mi: int,
    subject_id: str,
) -> tuple[np.ndarray, str]:
    """Compute a (3, 131, 131) subject tensor for a given build candidate."""
    run_ts: list[tuple[str, np.ndarray]] = []
    for _, row in run_rows.iterrows():
        sid = f"{subject_id}_run{row['run']}"
        ts = load_roi_timeseries(str(row["roi_txt_path"]), roi_mapping, sid)
        run_ts.append((str(row["run"]), ts))

    n_runs = len(run_ts)

    if candidate == "concatenated_timeseries":
        ts_cat = np.concatenate([ts for _, ts in run_ts], axis=0)
        raw = compute_raw_channels(ts_cat, n_jobs_mi)
        tensor = normalize_channels(raw)
        detail = f"concatenated {n_runs} run(s), total {ts_cat.shape[0]} timepoints"

    elif candidate == "runwise_140TR_connectome_average":
        per_run_raw: list[np.ndarray] = []
        for run_id, ts in run_ts:
            ts_trunc = ts[:TR_LIMIT_140, :]
            if ts_trunc.shape[0] < 2:
                raise ValueError(
                    f"{subject_id} run {run_id}: < 2 timepoints after {TR_LIMIT_140}TR truncation"
                )
            per_run_raw.append(compute_raw_channels(ts_trunc, n_jobs_mi))
        stacked = np.stack(per_run_raw, axis=0)     # (n_runs, 3, 131, 131)
        averaged = np.nanmean(stacked, axis=0)       # (3, 131, 131)
        tensor = normalize_channels(averaged)
        detail = f"averaged {n_runs} run-level raw connectome(s), first {TR_LIMIT_140} TPs each"

    elif candidate == "runwise164_connectome_average":
        per_run_raw = []
        for _, ts in run_ts:
            per_run_raw.append(compute_raw_channels(ts, n_jobs_mi))
        stacked = np.stack(per_run_raw, axis=0)
        averaged = np.nanmean(stacked, axis=0)
        tensor = normalize_channels(averaged)
        n_tp = run_ts[0][1].shape[0]
        detail = f"averaged {n_runs} run-level raw connectome(s), all {n_tp} TPs each"

    else:
        raise ValueError(f"Unknown build candidate: {candidate!r}")

    expected_shape = (len(CHANNEL_NAMES), EXPECTED_N_ROIS, EXPECTED_N_ROIS)
    if tensor.shape != expected_shape:
        raise ValueError(f"{subject_id}: tensor shape {tensor.shape} != expected {expected_shape}")

    return tensor, detail


# ---------------------------------------------------------------------------
# Tensor building loop
# ---------------------------------------------------------------------------

def build_tensor(
    candidate: str,
    subject_manifest: pd.DataFrame,
    run_table: pd.DataFrame,
    roi_mapping: pd.DataFrame,
    n_jobs_mi: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    runs_by_subject = {
        subj_id: grp.reset_index(drop=True)
        for subj_id, grp in run_table.groupby("subject_id")
    }

    tensors: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    n_total = len(subject_manifest)

    for n_done, (_, row) in enumerate(subject_manifest.iterrows(), start=1):
        subj_id = str(row["subject_id"])
        run_rows = runs_by_subject.get(subj_id)
        if run_rows is None or run_rows.empty:
            raise ValueError(f"No QC-passing runs for subject {subj_id!r}")

        diag = str(row["diagnosis"])
        print(
            f"  [{n_done:3d}/{n_total}] {subj_id} ({diag}) — {len(run_rows)} run(s)",
            flush=True,
        )

        subject_tensor, detail = build_subject_tensor(
            candidate, run_rows, roi_mapping, n_jobs_mi, subj_id
        )
        tensors.append(subject_tensor)
        qc_rows.append(
            {
                "subject_id": subj_id,
                "session_id": str(row.get("session_id", "")),
                "diagnosis": diag,
                "build_candidate": candidate,
                "n_runs_used": len(run_rows),
                "run_ids_used": ";".join(sorted(str(r) for r in run_rows["run"])),
                "strategy_detail": detail,
                "tensor_finite_fraction": float(np.isfinite(subject_tensor).mean()),
                "tensor_abs_max": float(np.nanmax(np.abs(subject_tensor))),
            }
        )

    global_tensor = np.stack(tensors, axis=0).astype(np.float32)
    return global_tensor, pd.DataFrame(qc_rows)


# ---------------------------------------------------------------------------
# Save / QC outputs
# ---------------------------------------------------------------------------

def save_tensor_npz(
    tensor: np.ndarray,
    subject_manifest: pd.DataFrame,
    qc_df: pd.DataFrame,
    candidate: str,
    output_dir: Path,
    roi_mapping: pd.DataFrame,
) -> Path:
    tensor_path = output_dir / f"tensor_{candidate}.npz"
    roi_names = roi_mapping["ADNI_final_ROI_name"].astype(str).to_numpy()

    np.savez_compressed(
        tensor_path,
        global_tensor_data=tensor,
        subject_ids=subject_manifest["subject_id"].astype(str).to_numpy(),
        session_ids=subject_manifest["session_id"].astype(str).to_numpy(),
        experiment_ids=subject_manifest["experiment_id"].astype(str).to_numpy(),
        diagnosis=subject_manifest["diagnosis"].astype(str).to_numpy(),
        channel_names=np.array(CHANNEL_NAMES, dtype=str),
        rois_count=np.array(EXPECTED_N_ROIS, dtype=np.int32),
        roi_order_name=np.array("aal3_adni_final_order", dtype=str),
        roi_names_in_order=roi_names.astype(str),
        build_candidate=np.array(candidate, dtype=str),
        oasis_input_dir=np.array(str(INPUT_ROOT), dtype=str),
        python_bandpass_applied=np.array(False),
        external_validation_only=np.array(True),
    )

    qc_csv = output_dir / f"tensor_qc_subjects_{candidate}.csv"
    qc_df.to_csv(qc_csv, index=False)
    return tensor_path


def plot_channel_distribution(
    tensor: np.ndarray, candidate: str, output_dir: Path
) -> None:
    n = EXPECTED_N_ROIS
    offdiag_mask = ~np.eye(n, dtype=bool)
    fig, axes = plt.subplots(1, len(CHANNEL_NAMES), figsize=(5 * len(CHANNEL_NAMES), 4),
                             sharey=False)

    for c, (ax, ch_name) in enumerate(zip(axes, CHANNEL_NAMES)):
        vals = tensor[:, c, :, :][:, offdiag_mask].ravel()
        ax.hist(vals, bins=100, density=True, alpha=0.75, color=f"C{c}")
        ax.set_title(ch_name.replace("_", " "), fontsize=8)
        ax.set_xlabel("Off-diagonal value (normalized)", fontsize=8)
        if c == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        stats_text = f"μ={vals.mean():.3f}\nσ={vals.std():.3f}\np5={np.percentile(vals, 5):.2f}\np95={np.percentile(vals, 95):.2f}"
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, family="monospace")

    fig.suptitle(
        f"Channel distribution — {candidate}\n(N={tensor.shape[0]} subjects, 131 ROIs)",
        fontsize=9,
    )
    fig.tight_layout()
    out_path = output_dir / f"channel_distribution_qc_{candidate}.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  QC figure: {out_path}", flush=True)


def summarize_qc(
    qc_tables: list[pd.DataFrame], tensor_paths: list[Path]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for qc_df, tensor_path in zip(qc_tables, tensor_paths):
        rows.append(
            {
                "build_candidate": qc_df["build_candidate"].iloc[0],
                "tensor_path": str(tensor_path),
                "n_subjects": int(len(qc_df)),
                "n_cn": int((qc_df["diagnosis"] == "CN").sum()),
                "n_ad_dementia": int((qc_df["diagnosis"] == "AD_DEMENTIA").sum()),
                "min_finite_fraction": float(qc_df["tensor_finite_fraction"].min()),
                "mean_finite_fraction": float(qc_df["tensor_finite_fraction"].mean()),
                "max_abs_value": float(qc_df["tensor_abs_max"].max()),
                "n_subjects_with_nonfinite": int((qc_df["tensor_finite_fraction"] < 1.0).sum()),
                "mean_runs_per_subject": float(qc_df["n_runs_used"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dry-run report
# ---------------------------------------------------------------------------

def write_dry_run_report(
    output_dir: Path,
    run_table: pd.DataFrame,
    subject_manifest: pd.DataFrame,
    candidates: list[str],
) -> None:
    n_cn = int((subject_manifest["diagnosis"] == "CN").sum())
    n_ad = int((subject_manifest["diagnosis"] == "AD_DEMENTIA").sum())
    n_runs = int(len(run_table))

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "mode": "dry_run",
        "confirm_build": False,
        "computed_connectomes": False,
        "modified_input_data": False,
        "no_training": True,
        "no_model_inference": True,
        "build_candidates": candidates,
        "channels": CHANNEL_NAMES,
        "n_neighbors_mi": N_NEIGHBORS_MI,
        "tr_limit_140tr_build": TR_LIMIT_140,
        "counts": {
            "selected_subjects": int(len(subject_manifest)),
            "cn": n_cn,
            "ad_dementia": n_ad,
            "qc_passing_runs": n_runs,
            "rois": EXPECTED_N_ROIS,
            "channels": len(CHANNEL_NAMES),
        },
        "guardrail": "real build requires --confirm-build",
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    readme = f"""# OASIS 60 CN / 60 AD Tensor Build — 2026-05-30

Mode: `dry_run` (no tensors computed — pass `--confirm-build` to build)

## Planned inputs

| Field | Value |
|---|---|
| Subjects | {len(subject_manifest)} (CN={n_cn}, AD_DEMENTIA={n_ad}) |
| QC-passing runs | {n_runs} |
| ROIs | {EXPECTED_N_ROIS} |
| Channels | {", ".join(CHANNEL_NAMES)} |
| Build candidates | {", ".join(candidates)} |
| MI n_neighbors | {N_NEIGHBORS_MI} |
| 140TR truncation | {TR_LIMIT_140} timepoints |

## To execute

```bash
/home/diego/anaconda3/envs/vae_ad/bin/python \\
    scripts/revision_bspc_2026/build_oasis_next_60cn_60ad_tensor_20260530.py \\
    --confirm-build \\
    --n-jobs-mi 8
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs ...", flush=True)
    process_ready, signal_qc, motion_qc, selected_ideal, roi_mapping = load_inputs()

    run_table = build_run_table(process_ready, signal_qc, motion_qc)
    subject_manifest = build_subject_manifest(process_ready, selected_ideal, run_table)

    candidates = _selected_candidates(args.build_candidate)

    print(f"Subjects  : {len(subject_manifest)}", flush=True)
    print(f"Runs      : {len(run_table)}", flush=True)
    print(f"Candidates: {candidates}", flush=True)

    # Always save planning artefacts
    write_csv_md(
        subject_manifest,
        args.output_dir / "subject_manifest.csv",
        args.output_dir / "subject_manifest.md",
        "OASIS 60 CN / 60 AD Subject Manifest",
    )
    run_display_cols = [
        c for c in ["subject_id", "bids_session", "run", "run_key",
                     "n_timepoints", "mean_fd_jenkinson"]
        if c in run_table.columns
    ]
    write_csv_md(
        run_table[run_display_cols],
        args.output_dir / "run_selection_used.csv",
        args.output_dir / "run_selection_used.md",
        "Run Selection Used",
    )
    roi_mapping.to_csv(args.output_dir / "roi_mapping_used.csv", index=False)

    if not args.confirm_build:
        write_dry_run_report(args.output_dir, run_table, subject_manifest, candidates)
        print(json.dumps({
            "output_dir": str(args.output_dir),
            "mode": "dry_run",
            "computed_connectomes": False,
        }, indent=2))
        return

    # Confirm-build path
    if args.fail_if_output_exists:
        for cand in candidates:
            tp = args.output_dir / f"tensor_{cand}.npz"
            if tp.exists():
                raise FileExistsError(f"Refusing to overwrite existing tensor: {tp}")

    qc_tables: list[pd.DataFrame] = []
    tensor_paths: list[Path] = []

    for candidate in candidates:
        print(f"\n=== Building: {candidate} ===", flush=True)
        tensor, qc_df = build_tensor(
            candidate, subject_manifest, run_table, roi_mapping, args.n_jobs_mi
        )
        tensor_path = save_tensor_npz(
            tensor, subject_manifest, qc_df, candidate, args.output_dir, roi_mapping
        )
        print(f"  Saved tensor: {tensor_path}  shape={tensor.shape}", flush=True)
        plot_channel_distribution(tensor, candidate, args.output_dir)
        qc_tables.append(qc_df)
        tensor_paths.append(tensor_path)

    qc_summary = summarize_qc(qc_tables, tensor_paths)
    write_csv_md(
        qc_summary,
        args.output_dir / "tensor_qc_summary.csv",
        args.output_dir / "tensor_qc_summary.md",
        "Tensor QC Summary",
    )

    log: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "mode": "build",
        "confirm_build": True,
        "computed_connectomes": True,
        "modified_input_data": False,
        "no_training": True,
        "no_model_inference": True,
        "n_jobs_mi": int(args.n_jobs_mi),
        "n_neighbors_mi": N_NEIGHBORS_MI,
        "tr_limit_140tr_build": TR_LIMIT_140,
        "build_candidates": candidates,
        "channels": CHANNEL_NAMES,
        "tensor_paths": [str(p) for p in tensor_paths],
        "counts": {
            "subjects": int(len(subject_manifest)),
            "cn": int((subject_manifest["diagnosis"] == "CN").sum()),
            "ad_dementia": int((subject_manifest["diagnosis"] == "AD_DEMENTIA").sum()),
            "qc_passing_runs": int(len(run_table)),
            "rois": EXPECTED_N_ROIS,
            "channels": len(CHANNEL_NAMES),
        },
    }
    (args.output_dir / "command_log.json").write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "mode": "build",
                "tensor_paths": [str(p) for p in tensor_paths],
                "computed_connectomes": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
