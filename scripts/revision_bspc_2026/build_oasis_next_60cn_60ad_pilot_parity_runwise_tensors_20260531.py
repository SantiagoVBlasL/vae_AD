#!/usr/bin/env python3
"""Build pilot-parity runwise OASIS tensors for the new 60CN/60AD batch.

This branch exists only to test the runwise tensor-construction mismatch found
by the pilot-vs-new parity audit. It rebuilds runwise tensors with the Tanda
pilot order:

1. load each QC-usable ROI time series run
2. compute run-level connectome channels
3. normalize each run-level channel off-diagonal values
4. average normalized run-level connectomes within subject/session
5. apply the same final per-channel off-diagonal normalization used by the
   pilot `combine_run_matrices` helper

Existing tensors are never overwritten unless an output path inside this new
branch already exists and `--force-overwrite-output-branch` is passed.
No scoring, training, or input-data modification is performed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.feature_selection import mutual_info_regression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_oasis_next_60cn_60ad_tensor_20260530 import (  # noqa: E402
    CHANNEL_NAMES,
    EXPECTED_N_ROIS,
    INPUT_ROOT,
    N_NEIGHBORS_MI,
    TR_LIMIT_140,
    build_run_table,
    build_subject_manifest,
    load_inputs,
    load_roi_timeseries,
    normalize_channels,
    pearson_full,
    plot_channel_distribution,
    write_csv_md,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531"
)
SOURCE_TENSOR_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_next_60cn_60ad_tensor_build_20260530"
)
BUILD_CANDIDATES = [
    "runwise_140TR_pilot_parity",
    "runwise164_pilot_parity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--build-candidate",
        choices=["all", *BUILD_CANDIDATES],
        default="all",
        help="Run one pilot-parity tensor or both.",
    )
    parser.add_argument("--n-jobs-mi", type=int, default=8)
    parser.add_argument("--confirm-build", action="store_true")
    parser.add_argument(
        "--force-overwrite-output-branch",
        action="store_true",
        help="Allow replacing tensor files inside this new output branch only.",
    )
    return parser.parse_args()


def selected_candidates(value: str) -> list[str]:
    return BUILD_CANDIDATES if value == "all" else [value]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def offdiag_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return mat[..., mask]


def finite_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def global_efficiency_wei_scipy(gw: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        lengths = np.where(gw > 0.0, 1.0 / gw, 0.0)
    dist = shortest_path(lengths, directed=False, unweighted=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.isfinite(dist) & (dist > 0.0), 1.0 / dist, 0.0)
    np.fill_diagonal(inv, 0.0)
    n = gw.shape[0]
    return float(inv.sum() / (n * n - n))


def threshold_omst_fast(mtx: np.ndarray, n_msts: int | None = None) -> np.ndarray:
    """Fast equivalent of dyconnmap threshold_omst_global_cost_efficiency.

    The original dyconnmap implementation uses NetworkX MSTs and a pure-Python
    BCT shortest-path loop. For this branch, SciPy is used for both operations.
    A benchmark in the command log confirms identical OMST masks for one OASIS
    run before the full build.
    """
    imtx = np.array(mtx, dtype=np.float64, copy=True)
    n = imtx.shape[0]
    imtx_up = np.array(mtx, dtype=np.float64, copy=True)
    imtx_up[np.tril_indices(n)] = 0.0
    np.fill_diagonal(imtx_up, 0.0)
    num_edges = len(np.where(imtx > 0.0)[0])
    if n_msts is None:
        num_msts = np.round(num_edges / (n - 1)) + 1
    else:
        num_msts = n_msts
    pos_num_msts = np.round(num_edges / (n - 1))
    if num_msts > pos_num_msts:
        num_msts = pos_num_msts
    num_msts = int(num_msts)

    cij_not = imtx.copy()
    cumulative = np.zeros((n, n), dtype=np.float64)
    cumulative_trees: list[np.ndarray] = []
    for _ in range(num_msts):
        with np.errstate(divide="ignore", invalid="ignore"):
            dist = np.where(cij_not > 0.0, 1.0 / cij_not, 0.0)
        mst = minimum_spanning_tree(dist).toarray()
        edges = np.argwhere(mst > 0.0)
        for i, j in edges:
            weight = imtx[i, j]
            cumulative[i, j] = weight
            cumulative[j, i] = weight
        cij_not = cij_not * (cumulative == 0.0)
        cumulative_trees.append(cumulative.copy())

    n_trees = np.stack(cumulative_trees, axis=0) if cumulative_trees else np.zeros((0, n, n), dtype=np.float64)
    global_eff_ini = global_efficiency_wei_scipy(imtx_up) * 2.0
    cost_ini = np.sum(imtx_up)
    gce = []
    for graph in n_trees:
        cost = np.sum(graph) / cost_ini
        ge = global_efficiency_wei_scipy(graph)
        gce.append(ge / global_eff_ini - cost)
    return n_trees[int(np.argmax(gce))].astype(np.float32)


def pearson_omst_fast(ts: np.ndarray) -> np.ndarray:
    z = pearson_full(ts)
    weights = np.abs(z)
    np.fill_diagonal(weights, 0.0)
    if np.all(np.isclose(weights, 0.0)):
        return z.astype(np.float32)
    omst = threshold_omst_fast(weights)
    out = z * (omst > 0.0).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def mi_knn_symmetric_targetwise(ts: np.ndarray, n_neighbors: int = N_NEIGHBORS_MI, n_jobs: int = 1) -> np.ndarray:
    n_tp, n_rois = ts.shape
    if n_tp <= n_neighbors:
        return np.zeros((n_rois, n_rois), dtype=np.float32)

    def one_target(j: int) -> np.ndarray:
        return mutual_info_regression(
            ts,
            ts[:, j],
            n_neighbors=n_neighbors,
            random_state=42,
            discrete_features=False,
        ).astype(np.float32)

    if n_jobs == 1:
        directed_cols = [one_target(j) for j in range(n_rois)]
    else:
        directed_cols = Parallel(n_jobs=n_jobs)(delayed(one_target)(j) for j in range(n_rois))
    directed = np.stack(directed_cols, axis=1).astype(np.float32)
    mat = (directed + directed.T) / 2.0
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)


def compute_run_channels_fast(ts: np.ndarray, n_jobs_mi: int) -> np.ndarray:
    raw = np.stack(
        [
            pearson_full(ts),
            pearson_omst_fast(ts),
            mi_knn_symmetric_targetwise(ts, n_jobs=n_jobs_mi),
        ],
        axis=0,
    ).astype(np.float32)
    return raw


def build_subject_tensor_pilot_parity(
    candidate: str,
    run_rows: pd.DataFrame,
    roi_mapping: pd.DataFrame,
    n_jobs_mi: int,
    subject_id: str,
) -> tuple[np.ndarray, str]:
    """Return a (3, 131, 131) subject tensor using pilot-compatible runwise order."""
    run_matrices: list[np.ndarray] = []
    original_lengths: list[int] = []
    for _, row in run_rows.iterrows():
        run_id = str(row["run"])
        sid = f"{subject_id}_run{run_id}"
        ts = load_roi_timeseries(str(row["roi_txt_path"]), roi_mapping, sid)
        original_lengths.append(int(ts.shape[0]))
        if candidate == "runwise_140TR_pilot_parity":
            ts = ts[:TR_LIMIT_140, :]
            if ts.shape[0] < 2:
                raise ValueError(f"{subject_id} run {run_id}: fewer than 2 TR after {TR_LIMIT_140}TR truncation")
        elif candidate == "runwise164_pilot_parity":
            pass
        else:
            raise ValueError(f"Unknown candidate: {candidate}")

        # Pilot-compatible order: compute raw channels for one run, then
        # per-run normalize channels before any run averaging.
        per_run_raw = compute_run_channels_fast(ts, n_jobs_mi)
        per_run_norm = normalize_channels(per_run_raw)
        run_matrices.append(per_run_norm)

    stacked = np.stack(run_matrices, axis=0).astype(np.float32)
    averaged_normalized = np.nanmean(stacked, axis=0).astype(np.float32)
    # Matches pilot combine_run_matrices: final off-diagonal normalization after
    # averaging already-normalized run-level connectomes.
    tensor = normalize_channels(averaged_normalized)
    detail = (
        f"pilot-parity: normalized {len(run_matrices)} run-level connectome(s), "
        f"averaged normalized runs, final normalized average; original_lengths={';'.join(map(str, original_lengths))}"
    )
    expected_shape = (len(CHANNEL_NAMES), EXPECTED_N_ROIS, EXPECTED_N_ROIS)
    if tensor.shape != expected_shape:
        raise ValueError(f"{subject_id}: tensor shape {tensor.shape} != expected {expected_shape}")
    return tensor.astype(np.float32), detail


def build_tensor(
    candidate: str,
    subject_manifest: pd.DataFrame,
    run_table: pd.DataFrame,
    roi_mapping: pd.DataFrame,
    n_jobs_mi: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    runs_by_subject = {sid: grp.reset_index(drop=True) for sid, grp in run_table.groupby("subject_id")}
    tensors: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    n_total = len(subject_manifest)

    for n_done, (_, row) in enumerate(subject_manifest.iterrows(), start=1):
        sid = str(row["subject_id"])
        run_rows = runs_by_subject.get(sid)
        if run_rows is None or run_rows.empty:
            raise ValueError(f"No QC-passing runs for subject {sid!r}")
        print(f"  [{n_done:3d}/{n_total}] {sid} ({row['diagnosis']}) — {len(run_rows)} run(s)", flush=True)
        tensor, detail = build_subject_tensor_pilot_parity(candidate, run_rows, roi_mapping, n_jobs_mi, sid)
        tensors.append(tensor)
        qc_rows.append(
            {
                "subject_id": sid,
                "session_id": str(row.get("session_id", "")),
                "diagnosis": str(row["diagnosis"]),
                "build_candidate": candidate,
                "n_runs_used": len(run_rows),
                "run_ids_used": ";".join(sorted(str(r) for r in run_rows["run"])),
                "strategy_detail": detail,
                "tensor_finite_fraction": float(np.isfinite(tensor).mean()),
                "tensor_abs_max": float(np.nanmax(np.abs(tensor))),
            }
        )
    return np.stack(tensors, axis=0).astype(np.float32), pd.DataFrame(qc_rows)


def save_tensor_npz(
    tensor: np.ndarray,
    subject_manifest: pd.DataFrame,
    candidate: str,
    output_dir: Path,
    roi_mapping: pd.DataFrame,
) -> Path:
    path = output_dir / f"tensor_{candidate}.npz"
    roi_names = roi_mapping["ADNI_final_ROI_name"].astype(str).to_numpy()
    np.savez_compressed(
        path,
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
        runwise_build_order=np.array(
            "per_run_compute_then_normalize__average_normalized_runs__final_normalize_average",
            dtype=str,
        ),
        oasis_input_dir=np.array(str(INPUT_ROOT), dtype=str),
        source_tensor_branch=np.array(str(SOURCE_TENSOR_DIR), dtype=str),
        python_bandpass_applied=np.array(False),
        external_validation_only=np.array(True),
    )
    return path


def tensor_qc_rows(tensor: np.ndarray, candidate: str, tensor_path: Path, subject_manifest: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    n_subjects, n_channels, n_roi_a, n_roi_b = tensor.shape
    diag = np.diagonal(tensor, axis1=-2, axis2=-1)
    sym = np.abs(tensor - np.swapaxes(tensor, -1, -2))
    diag_stats = finite_stats(diag)
    sym_stats = finite_stats(sym)
    diagnosis_counts = subject_manifest["diagnosis"].value_counts(dropna=False).to_dict()
    summary = [
        {
            "build_candidate": candidate,
            "tensor_path": str(tensor_path),
            "shape": str(tuple(tensor.shape)),
            "subject_count": int(n_subjects),
            "channel_count": int(n_channels),
            "roi_count_a": int(n_roi_a),
            "roi_count_b": int(n_roi_b),
            "channel_names": ";".join(CHANNEL_NAMES),
            "n_cn": int(diagnosis_counts.get("CN", 0)),
            "n_ad_dementia": int(diagnosis_counts.get("AD_DEMENTIA", 0)),
            "nan_count": int(np.isnan(tensor).sum()),
            "inf_count": int(np.isinf(tensor).sum()),
            "diagonal_mean": diag_stats["mean"],
            "diagonal_std": diag_stats["std"],
            "diagonal_min": diag_stats["min"],
            "diagonal_max": diag_stats["max"],
            "diagonal_abs_max": float(np.nanmax(np.abs(diag))),
            "symmetry_abs_mean": sym_stats["mean"],
            "symmetry_abs_max": sym_stats["max"],
            "ready_for_scoring": bool(
                tensor.shape == (120, 3, EXPECTED_N_ROIS, EXPECTED_N_ROIS)
                and diagnosis_counts.get("CN", 0) == 60
                and diagnosis_counts.get("AD_DEMENTIA", 0) == 60
                and int(np.isnan(tensor).sum()) == 0
                and int(np.isinf(tensor).sum()) == 0
                and float(np.nanmax(np.abs(diag))) <= 1e-5
                and float(np.nanmax(sym)) <= 1e-5
            ),
        }
    ]
    channel_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(CHANNEL_NAMES):
        vals = tensor[:, idx, :, :]
        off = offdiag_values(vals)
        all_stats = finite_stats(vals)
        off_stats = finite_stats(off)
        diag_ch = np.diagonal(vals, axis1=-2, axis2=-1)
        sym_ch = np.abs(vals - np.swapaxes(vals, -1, -2))
        channel_rows.append(
            {
                "build_candidate": candidate,
                "channel_index": idx,
                "channel_name": name,
                "all_mean": all_stats["mean"],
                "all_std": all_stats["std"],
                "all_min": all_stats["min"],
                "all_max": all_stats["max"],
                "offdiag_mean": off_stats["mean"],
                "offdiag_std": off_stats["std"],
                "offdiag_min": off_stats["min"],
                "offdiag_max": off_stats["max"],
                "diagonal_abs_max": float(np.nanmax(np.abs(diag_ch))),
                "symmetry_abs_max": float(np.nanmax(sym_ch)),
                "nan_count": int(np.isnan(vals).sum()),
                "inf_count": int(np.isinf(vals).sum()),
            }
        )
    diagnosis_rows = [
        {"build_candidate": candidate, "diagnosis": str(dx), "n": int(n)}
        for dx, n in sorted(diagnosis_counts.items())
    ]
    return summary, channel_rows, diagnosis_rows


def write_readme(output_dir: Path, built: bool, candidates: list[str]) -> None:
    text = f"""# OASIS Next 60CN/60AD Pilot-Parity Runwise Tensor Build

Mode: `{"build_complete" if built else "dry_run"}`

This branch rebuilds only runwise OASIS tensors with the Tanda pilot-compatible
order:

1. compute connectome channels per QC-usable run,
2. normalize each run-level channel off-diagonal values,
3. average normalized run-level connectomes per subject/session,
4. apply the same final per-channel normalization used by the pilot helper.

Existing 20260530 tensors are not overwritten.

## Candidates

{chr(10).join(f"- `{c}`" for c in candidates)}

## Guardrails

- No scoring.
- No model training.
- No input-data modification.
- No modification of previous tensor outputs.
"""
    output_dir.joinpath("README.md").write_text(text, encoding="utf-8")


def dry_run(args: argparse.Namespace, run_table: pd.DataFrame, subject_manifest: pd.DataFrame, candidates: list[str]) -> None:
    write_readme(args.output_dir, built=False, candidates=candidates)
    write_json(
        args.output_dir / "command_log.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "mode": "dry_run",
            "confirm_build": False,
            "computed_connectomes": False,
            "modified_input_data": False,
            "modified_previous_tensors": False,
            "scoring_launched": False,
            "training_launched": False,
            "build_candidates": candidates,
            "counts": {
                "subjects": int(len(subject_manifest)),
                "cn": int((subject_manifest["diagnosis"] == "CN").sum()),
                "ad_dementia": int((subject_manifest["diagnosis"] == "AD_DEMENTIA").sum()),
                "qc_passing_runs": int(len(run_table)),
                "channels": int(len(CHANNEL_NAMES)),
                "rois": int(EXPECTED_N_ROIS),
            },
        },
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = selected_candidates(args.build_candidate)

    process_ready, signal_qc, motion_qc, selected_ideal, roi_mapping = load_inputs()
    run_table = build_run_table(process_ready, signal_qc, motion_qc)
    subject_manifest = build_subject_manifest(process_ready, selected_ideal, run_table)

    write_csv_md(subject_manifest, args.output_dir / "subject_manifest.csv", args.output_dir / "subject_manifest.md", "Subject Manifest")
    run_cols = [
        c
        for c in ["subject_id", "bids_session", "run", "run_key", "n_timepoints", "mean_fd_jenkinson", "roi_txt_path"]
        if c in run_table.columns
    ]
    write_csv_md(run_table[run_cols], args.output_dir / "run_selection_used.csv", args.output_dir / "run_selection_used.md", "Run Selection Used")
    write_csv_md(roi_mapping, args.output_dir / "roi_mapping_used.csv", args.output_dir / "roi_mapping_used.md", "ROI Mapping Used")

    if not args.confirm_build:
        dry_run(args, run_table, subject_manifest, candidates)
        print(json.dumps({"output_dir": str(args.output_dir), "mode": "dry_run", "computed_connectomes": False}, indent=2))
        return 0

    for candidate in candidates:
        tensor_path = args.output_dir / f"tensor_{candidate}.npz"
        if tensor_path.exists() and not args.force_overwrite_output_branch:
            raise FileExistsError(
                f"Refusing to overwrite existing tensor in output branch: {tensor_path}. "
                "Use --force-overwrite-output-branch only for this branch."
            )

    all_summary: list[dict[str, Any]] = []
    all_channels: list[dict[str, Any]] = []
    all_diagnosis: list[dict[str, Any]] = []
    tensor_paths: list[Path] = []
    for candidate in candidates:
        print(f"\n=== Building pilot-parity tensor: {candidate} ===", flush=True)
        tensor, qc_df = build_tensor(candidate, subject_manifest, run_table, roi_mapping, int(args.n_jobs_mi))
        tensor_path = save_tensor_npz(tensor, subject_manifest, candidate, args.output_dir, roi_mapping)
        tensor_paths.append(tensor_path)
        qc_df.to_csv(args.output_dir / f"tensor_qc_subjects_{candidate}.csv", index=False)
        plot_channel_distribution(tensor, candidate, args.output_dir)
        summary, channel_rows, diagnosis_rows = tensor_qc_rows(tensor, candidate, tensor_path, subject_manifest)
        all_summary.extend(summary)
        all_channels.extend(channel_rows)
        all_diagnosis.extend(diagnosis_rows)
        print(f"Saved: {tensor_path} shape={tensor.shape}", flush=True)

    summary_df = pd.DataFrame(all_summary)
    channel_df = pd.DataFrame(all_channels)
    diagnosis_df = pd.DataFrame(all_diagnosis)
    write_csv_md(summary_df, args.output_dir / "tensor_qc_summary.csv", args.output_dir / "tensor_qc_summary.md", "Tensor QC Summary")
    write_csv_md(channel_df, args.output_dir / "channel_statistics.csv", args.output_dir / "channel_statistics.md", "Channel Statistics")
    write_csv_md(diagnosis_df, args.output_dir / "diagnosis_counts.csv", args.output_dir / "diagnosis_counts.md", "Diagnosis Counts")

    decision = "ready_for_scoring" if bool(summary_df["ready_for_scoring"].all()) and len(summary_df) == len(candidates) else "not_ready"
    final_text = f"""# Final Recommendation

Decision: `{decision}`

Built pilot-parity runwise tensors in a new output branch. Existing 20260530 tensors were not overwritten.

No scoring, model training, input-data modification, or previous tensor modification was performed.
"""
    args.output_dir.joinpath("final_recommendation.md").write_text(final_text, encoding="utf-8")
    write_readme(args.output_dir, built=True, candidates=candidates)
    write_json(
        args.output_dir / "command_log.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": str(Path(__file__).resolve()),
            "mode": "build",
            "confirm_build": True,
            "computed_connectomes": True,
            "modified_input_data": False,
            "modified_previous_tensors": False,
            "scoring_launched": False,
            "training_launched": False,
            "n_jobs_mi": int(args.n_jobs_mi),
            "n_neighbors_mi": int(N_NEIGHBORS_MI),
            "build_candidates": candidates,
            "tensor_paths": [str(p) for p in tensor_paths],
            "recommendation": decision,
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "mode": "build", "recommendation": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
