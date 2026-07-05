#!/usr/bin/env python
"""Build OASIS Tanda 2026-05-25 connectomes after ROI-order confirmation.

Default behavior is a read-only/dry-run execution package. Real tensor building
requires both:

1. --confirm-build
2. either --confirm-roi-order-aal3-1to170 or the file
   results/revision_bspc_2026/oasis_tanda_2026_05_25_roi_mapping_audit/
   martin_roi_order_confirmation.md

The script never modifies input OASIS data.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Tanda_2026_05_25"
DEFAULT_PREFLIGHT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectome_build_preflight"
)
DEFAULT_ROI_MAPPING_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_roi_mapping_audit"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectomes"
)
CONFIRMATION_FILENAME = "martin_roi_order_confirmation.md"

CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
BUILD_CANDIDATES = ["concatenated_timeseries", "runwise_connectome_average"]
N_NEIGHBORS_MI = 5


@dataclass(frozen=True)
class BuildContext:
    output_dir: Path
    run_selection: pd.DataFrame
    subject_manifest: pd.DataFrame
    roi_mapping: pd.DataFrame
    roi_order_confirmed: bool
    roi_order_confirmation_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    parser.add_argument("--roi-mapping-audit-dir", type=Path, default=DEFAULT_ROI_MAPPING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--build-candidate",
        choices=["both", *BUILD_CANDIDATES],
        default="both",
        help="Which tensor candidate to build when build is confirmed.",
    )
    parser.add_argument("--confirm-build", action="store_true")
    parser.add_argument("--confirm-roi-order-aal3-1to170", action="store_true")
    parser.add_argument("--n-jobs-mi", type=int, default=1)
    parser.add_argument("--fail-if-output-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run alias; default without --confirm-build.")
    return parser.parse_args()


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


def require_inputs(args: argparse.Namespace) -> None:
    required = [
        args.input_dir,
        args.preflight_dir / "run_selection_table.csv",
        args.preflight_dir / "subject_level_manifest.csv",
        args.preflight_dir / "roi_mapping_used.csv",
        args.roi_mapping_audit_dir,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))


def roi_order_confirmation(args: argparse.Namespace) -> tuple[bool, str]:
    confirmation_file = args.roi_mapping_audit_dir / CONFIRMATION_FILENAME
    if args.confirm_roi_order_aal3_1to170:
        return True, "--confirm-roi-order-aal3-1to170"
    if confirmation_file.exists():
        return True, str(confirmation_file)
    return False, "missing explicit ROI-order confirmation"


def load_context(args: argparse.Namespace) -> BuildContext:
    require_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_selection = pd.read_csv(args.preflight_dir / "run_selection_table.csv")
    subject_manifest = pd.read_csv(args.preflight_dir / "subject_level_manifest.csv")
    roi_mapping = pd.read_csv(args.preflight_dir / "roi_mapping_used.csv")
    if len(roi_mapping) != 131:
        raise ValueError(f"Expected 131 ROI mapping rows, got {len(roi_mapping)}")
    if not roi_mapping["matches_adni_final_order"].astype(bool).all():
        raise ValueError("ROI mapping is not in exact ADNI final order")
    confirmed, source = roi_order_confirmation(args)
    return BuildContext(
        output_dir=args.output_dir,
        run_selection=run_selection,
        subject_manifest=subject_manifest,
        roi_mapping=roi_mapping,
        roi_order_confirmed=confirmed,
        roi_order_confirmation_source=source,
    )


def selected_candidates(build_candidate: str) -> list[str]:
    return BUILD_CANDIDATES if build_candidate == "both" else [build_candidate]


def selected_run_rows(run_selection: pd.DataFrame) -> pd.DataFrame:
    selected = run_selection[run_selection["selected_for_preflight"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("No QC-usable OASIS runs selected for build")
    return selected.sort_values(["subject_id", "session_id", "run_id"]).reset_index(drop=True)


def validate_real_build_allowed(args: argparse.Namespace, context: BuildContext) -> bool:
    if not args.confirm_build:
        return False
    if not context.roi_order_confirmed:
        raise RuntimeError(
            "Refusing to compute OASIS connectomes: ROI order is not externally confirmed. "
            "Provide --confirm-roi-order-aal3-1to170 or create "
            f"{context.output_dir.parent / 'oasis_tanda_2026_05_25_roi_mapping_audit' / CONFIRMATION_FILENAME}."
        )
    for candidate in selected_candidates(args.build_candidate):
        tensor_path = context.output_dir / f"tensor_{candidate}.npz"
        if args.fail_if_output_exists and tensor_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing tensor: {tensor_path}")
    return True


def copy_planning_outputs(context: BuildContext) -> None:
    run_selection = context.run_selection.copy()
    run_selection["roi_order_confirmation_required"] = True
    run_selection["roi_order_confirmed"] = context.roi_order_confirmed
    run_selection["roi_order_confirmation_source"] = context.roi_order_confirmation_source
    subject_manifest = context.subject_manifest.copy()
    subject_manifest["roi_order_confirmed"] = context.roi_order_confirmed
    subject_manifest["planned_channels"] = ";".join(CHANNEL_NAMES)
    write_csv_md(
        subject_manifest,
        context.output_dir / "subject_manifest.csv",
        context.output_dir / "subject_manifest.md",
        "OASIS Subject Manifest For Connectome Build",
    )
    write_csv_md(
        run_selection,
        context.output_dir / "run_selection_used.csv",
        context.output_dir / "run_selection_used.md",
        "Run Selection Used",
    )
    write_csv_md(
        context.roi_mapping,
        context.output_dir / "roi_mapping_used.csv",
        context.output_dir / "roi_mapping_used.md",
        "ROI Mapping Used",
    )


def load_roi_timeseries(path: str, roi_mapping: pd.DataFrame, sid: str) -> np.ndarray:
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Missing ROI time-series file for {sid}: {txt_path}")
    matrix = np.loadtxt(txt_path, delimiter=",")
    if matrix.ndim != 2:
        raise ValueError(f"{sid}: ROI signal matrix is not 2D: shape={matrix.shape}")
    if matrix.shape[1] == 170:
        ts_170 = matrix
    elif matrix.shape[0] == 170:
        ts_170 = matrix.T
    else:
        raise ValueError(f"{sid}: expected one dimension with 170 AAL3 columns, got {matrix.shape}")
    cols = roi_mapping["oasis_col_idx_0based"].astype(int).to_numpy()
    ts_131 = ts_170[:, cols].astype(np.float32)
    if ts_131.shape[1] != 131:
        raise ValueError(f"{sid}: retained ROI shape mismatch: {ts_131.shape}")
    all_nan_cols = np.where(np.isnan(ts_131).all(axis=0))[0].tolist()
    if all_nan_cols:
        raise ValueError(f"{sid}: retained ADNI ROI columns are all-NaN after mapping: {all_nan_cols}")
    if not np.isfinite(ts_131).all():
        ts_131 = np.nan_to_num(ts_131, nan=np.nanmedian(ts_131), posinf=0.0, neginf=0.0)
    return ts_131


def fisher_r_to_z(r_matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r_clean = np.nan_to_num(r_matrix.astype(np.float32), nan=0.0)
    r_clipped = np.clip(r_clean, -1.0 + eps, 1.0 - eps)
    z_matrix = np.arctanh(r_clipped)
    np.fill_diagonal(z_matrix, 0.0)
    return z_matrix.astype(np.float32)


def pearson_full(ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ts, rowvar=False).astype(np.float32)
    return fisher_r_to_z(corr)


def pearson_omst(ts: np.ndarray) -> np.ndarray:
    try:
        from dyconnmap.graphs.threshold import threshold_omst_global_cost_efficiency
    except Exception as exc:  # pragma: no cover - dependency checked only during real build
        raise RuntimeError("dyconnmap OMST dependency is unavailable") from exc

    z = pearson_full(ts)
    weights = np.abs(z)
    np.fill_diagonal(weights, 0.0)
    if np.all(np.isclose(weights, 0)):
        return z.astype(np.float32)
    outputs = threshold_omst_global_cost_efficiency(weights, n_msts=None)
    if not (isinstance(outputs, tuple) and len(outputs) >= 2):
        raise RuntimeError(f"Unexpected OMST return value: {type(outputs)}")
    omst_weighted = np.asarray(outputs[1]).astype(np.float32)
    mask = (omst_weighted > 0).astype(np.float32)
    out = z * mask
    np.fill_diagonal(out, 0.0)
    return out.astype(np.float32)


def _mi_pair(x: np.ndarray, y: np.ndarray, n_neighbors: int) -> float:
    try:
        return float(
            mutual_info_regression(
                x.reshape(-1, 1),
                y,
                n_neighbors=n_neighbors,
                random_state=42,
                discrete_features=False,
            )[0]
        )
    except Exception:
        return 0.0


def mi_knn_symmetric(ts: np.ndarray, n_neighbors: int = N_NEIGHBORS_MI, n_jobs: int = 1) -> np.ndarray:
    from joblib import Parallel, delayed

    n_tp, n_rois = ts.shape
    if n_tp <= n_neighbors:
        return np.zeros((n_rois, n_rois), dtype=np.float32)
    pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]
    if n_jobs == 1:
        vals = [(_mi_pair(ts[:, i], ts[:, j], n_neighbors), _mi_pair(ts[:, j], ts[:, i], n_neighbors)) for i, j in pairs]
    else:
        vals_ij = Parallel(n_jobs=n_jobs)(
            delayed(_mi_pair)(ts[:, i], ts[:, j], n_neighbors) for i, j in pairs
        )
        vals_ji = Parallel(n_jobs=n_jobs)(
            delayed(_mi_pair)(ts[:, j], ts[:, i], n_neighbors) for i, j in pairs
        )
        vals = list(zip(vals_ij, vals_ji))
    mat = np.zeros((n_rois, n_rois), dtype=np.float32)
    for (i, j), (ij, ji) in zip(pairs, vals):
        mat[i, j] = mat[j, i] = (ij + ji) / 2.0
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


def compute_channels(ts: np.ndarray, n_jobs_mi: int) -> np.ndarray:
    matrices = [
        pearson_full(ts),
        pearson_omst(ts),
        mi_knn_symmetric(ts, n_jobs=n_jobs_mi),
    ]
    normalized = [normalize_channel_offdiag(m) for m in matrices]
    return np.stack(normalized, axis=0).astype(np.float32)


def combine_run_matrices(run_matrices: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(run_matrices), axis=0).astype(np.float32)
    averaged = np.nanmean(stacked, axis=0)
    return np.stack([normalize_channel_offdiag(averaged[i]) for i in range(averaged.shape[0])], axis=0)


def build_tensor_for_candidate(
    candidate: str,
    context: BuildContext,
    n_jobs_mi: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    selected = selected_run_rows(context.run_selection)
    subjects = context.subject_manifest[context.subject_manifest["planned_include_subject_session"].astype(bool)].copy()
    subjects = subjects.sort_values(["subject_id", "session_id"]).reset_index(drop=True)
    tensors: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    selected_by_session = {
        key: group.sort_values("run_id")
        for key, group in selected.groupby(["subject_id", "session_id"], dropna=False)
    }

    for _, subj in subjects.iterrows():
        key = (subj["subject_id"], subj["session_id"])
        run_group = selected_by_session.get(key)
        if run_group is None or run_group.empty:
            raise ValueError(f"No selected runs for included subject/session {key}")
        run_ts = []
        for _, run in run_group.iterrows():
            sid = f"{run['subject_id']}_{run['session_id']}_{run['run_id']}"
            ts = load_roi_timeseries(str(run["roi_txt_path"]), context.roi_mapping, sid)
            run_ts.append((run["run_id"], ts))

        if candidate == "concatenated_timeseries":
            ts_concat = np.concatenate([ts for _, ts in run_ts], axis=0)
            subject_tensor = compute_channels(ts_concat, n_jobs_mi)
            strategy_detail = f"concatenated {len(run_ts)} run(s), {ts_concat.shape[0]} timepoints"
        elif candidate == "runwise_connectome_average":
            run_matrices = [compute_channels(ts, n_jobs_mi) for _, ts in run_ts]
            subject_tensor = combine_run_matrices(run_matrices)
            strategy_detail = f"averaged {len(run_ts)} run-level connectome(s)"
        else:
            raise ValueError(f"Unknown build candidate: {candidate}")

        if subject_tensor.shape != (len(CHANNEL_NAMES), 131, 131):
            raise ValueError(f"{key}: tensor shape mismatch {subject_tensor.shape}")
        tensors.append(subject_tensor)
        qc_rows.append(
            {
                "subject_id": subj["subject_id"],
                "session_id": subj["session_id"],
                "experiment_id": subj["experiment_id"],
                "build_candidate": candidate,
                "n_runs_used": len(run_ts),
                "run_ids_used": ";".join(str(rid) for rid, _ in run_ts),
                "strategy_detail": strategy_detail,
                "tensor_finite_fraction": float(np.isfinite(subject_tensor).mean()),
                "tensor_abs_max": float(np.nanmax(np.abs(subject_tensor))),
            }
        )

    return np.stack(tensors, axis=0).astype(np.float32), subjects, pd.DataFrame(qc_rows)


def save_tensor_npz(
    tensor: np.ndarray,
    subjects: pd.DataFrame,
    qc_df: pd.DataFrame,
    candidate: str,
    context: BuildContext,
) -> Path:
    tensor_path = context.output_dir / f"tensor_{candidate}.npz"
    roi_names = context.roi_mapping["ADNI_final_ROI_name"].astype(str).to_numpy()
    network_labels = np.array(["unknown_external_oasis"] * len(roi_names), dtype=str)
    np.savez_compressed(
        tensor_path,
        global_tensor_data=tensor,
        subject_ids=subjects["subject_id"].astype(str).to_numpy(),
        session_ids=subjects["session_id"].astype(str).to_numpy(),
        experiment_ids=subjects["experiment_id"].astype(str).to_numpy(),
        diagnosis=subjects["diagnosis"].astype(str).to_numpy(),
        channel_names=np.array(CHANNEL_NAMES, dtype=str),
        rois_count=np.array(131, dtype=np.int32),
        roi_order_name=np.array("aal3_manual_yeo17_order", dtype=str),
        roi_names_in_order=roi_names.astype(str),
        network_labels_in_order=network_labels,
        build_candidate=np.array(candidate, dtype=str),
        oasis_input_dir=np.array(str(DEFAULT_INPUT_DIR), dtype=str),
        python_bandpass_applied=np.array(False),
        external_validation_only=np.array(True),
    )
    qc_df.to_csv(context.output_dir / f"tensor_qc_subjects_{candidate}.csv", index=False)
    return tensor_path


def summarize_tensor_qc(qc_tables: list[pd.DataFrame], tensor_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for qc_df, tensor_path in zip(qc_tables, tensor_paths):
        rows.append(
            {
                "build_candidate": qc_df["build_candidate"].iloc[0],
                "tensor_path": str(tensor_path),
                "n_subject_sessions": int(len(qc_df)),
                "min_finite_fraction": float(qc_df["tensor_finite_fraction"].min()),
                "mean_finite_fraction": float(qc_df["tensor_finite_fraction"].mean()),
                "max_abs_value": float(qc_df["tensor_abs_max"].max()),
                "n_subjects_with_nonfinite": int((qc_df["tensor_finite_fraction"] < 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_blocked_report(args: argparse.Namespace, context: BuildContext, will_build: bool) -> None:
    selected_runs = selected_run_rows(context.run_selection)
    included = context.subject_manifest[context.subject_manifest["planned_include_subject_session"].astype(bool)]
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "mode": "build" if will_build else "dry_run_blocked_or_unconfirmed",
        "confirm_build": bool(args.confirm_build),
        "roi_order_confirmed": bool(context.roi_order_confirmed),
        "roi_order_confirmation_source": context.roi_order_confirmation_source,
        "computed_connectomes": bool(will_build),
        "modified_input_data": False,
        "build_candidates_requested": selected_candidates(args.build_candidate),
        "channels": CHANNEL_NAMES,
        "counts": {
            "selected_qc_runs": int(len(selected_runs)),
            "included_subject_sessions": int(len(included)),
            "rois": int(len(context.roi_mapping)),
            "channels": int(len(CHANNEL_NAMES)),
        },
        "guardrail": (
            "real build requires --confirm-build plus --confirm-roi-order-aal3-1to170 "
            f"or {CONFIRMATION_FILENAME}"
        ),
    }
    (context.output_dir / "command_log.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    dry_text = f"""# OASIS Connectome Build Execution Package

Mode: `{"build" if will_build else "dry_run_blocked_or_unconfirmed"}`

## Guardrail Status

- `--confirm-build`: {bool(args.confirm_build)}
- ROI order confirmed: {bool(context.roi_order_confirmed)}
- ROI confirmation source: `{context.roi_order_confirmation_source}`
- Connectomes computed: {bool(will_build)}

Real tensor construction is blocked unless both the build flag and ROI-order confirmation are present.

## Planned Inputs

- Selected QC-usable runs: {len(selected_runs)}
- Included subject/sessions: {len(included)}
- ROI count: {len(context.roi_mapping)}
- Channels: `{", ".join(CHANNEL_NAMES)}`
- Build candidates requested: `{", ".join(selected_candidates(args.build_candidate))}`

## Confirmation Requirement

Martin must confirm that `ROISignals_AAL3_FunImgARWSDCFN` columns are AAL3 label/color order 1..170, including empty columns 35, 36, 81, and 82.

## Future Build Command

After confirmation, either create `{CONFIRMATION_FILENAME}` in the ROI mapping audit folder or pass the explicit confirmation flag:

```bash
/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/build_oasis_tanda_20260525_connectomes.py --confirm-build --confirm-roi-order-aal3-1to170
```

Then audit the tensors:

```bash
/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/audit_oasis_tanda_20260525_connectome_tensor.py
```
"""
    (context.output_dir / "build_execution_status.md").write_text(dry_text, encoding="utf-8")
    (context.output_dir / "README.md").write_text(dry_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    context = load_context(args)
    will_build = validate_real_build_allowed(args, context)
    copy_planning_outputs(context)
    write_blocked_report(args, context, will_build)

    if not will_build:
        print(
            json.dumps(
                {
                    "output_dir": str(context.output_dir),
                    "mode": "dry_run_blocked_or_unconfirmed",
                    "confirm_build": bool(args.confirm_build),
                    "roi_order_confirmed": bool(context.roi_order_confirmed),
                    "computed_connectomes": False,
                },
                indent=2,
            )
        )
        return

    qc_tables: list[pd.DataFrame] = []
    tensor_paths: list[Path] = []
    for candidate in selected_candidates(args.build_candidate):
        tensor, subjects, qc_df = build_tensor_for_candidate(candidate, context, args.n_jobs_mi)
        tensor_path = save_tensor_npz(tensor, subjects, qc_df, candidate, context)
        qc_tables.append(qc_df)
        tensor_paths.append(tensor_path)

    tensor_qc_summary = summarize_tensor_qc(qc_tables, tensor_paths)
    write_csv_md(
        tensor_qc_summary,
        context.output_dir / "tensor_qc_summary.csv",
        context.output_dir / "tensor_qc_summary.md",
        "Tensor QC Summary",
    )

    write_blocked_report(args, context, will_build=True)
    print(
        json.dumps(
            {
                "output_dir": str(context.output_dir),
                "mode": "build",
                "tensor_paths": [str(p) for p in tensor_paths],
                "computed_connectomes": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
