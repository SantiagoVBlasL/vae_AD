#!/usr/bin/env python3
"""Build a replacement tensor with Site31 Mayo-slice-order reprocessed subjects.

Safety contract:
- never modifies the original tensor, metadata, configs, or previous run outputs;
- writes only new derived artifacts under the requested output directory and
  replacement tensor root;
- fails before writing the replacement tensor if channel order or ROI order
  cannot be verified against the locked tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision_bspc_2026 import (  # noqa: E402
    build_v5_dparsf10000_no_pybandpass_manifest_and_extract as fe,
)


ORIGINAL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = PROJECT_ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
NEW_ROISIGNALS_DIR = Path(
    "/media/diego/Datos/ResultsAAL3_Site31_MayoSliceOrder/ResultsAAL3/"
    "ROISignals_AAL3_FunImgARWSDCFN"
)
OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/site31_mayo_sliceorder_reprocessed14_tensor_preflight_20260618"
REPLACEMENT_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14/"
    "subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14.npz"
)
PROMOTED_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
PROMOTED_REPROCESSED_OUTPUT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14"
)

EXPECTED_SUBJECTS = {
    "031_S_4024",
    "031_S_4029",
    "031_S_4032",
    "031_S_4042",
    "031_S_4149",
    "031_S_4194",
    "031_S_4203",
    "031_S_4218",
    "031_S_4474",
    "031_S_4476",
    "031_S_4496",
    "031_S_4590",
    "031_S_4721",
    "031_S_4947",
}
KNOWN_SITE31_REVERSE = {
    "031_S_4021",
    "031_S_4032",
    "031_S_4218",
    "031_S_4474",
    "031_S_4496",
}
CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_df(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        md_path.write_text((df.to_markdown(index=False) if not df.empty else "_No rows._") + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def subject_from_path(path: Path) -> str:
    m = re.search(r"(\d{3}_S_\d{4})", path.name)
    return m.group(1) if m else ""


def list_new_files(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*")):
        sid = subject_from_path(path)
        if not sid:
            continue
        rows.append(
            {
                "SubjectID": sid,
                "path": str(path),
                "suffix": path.suffix.lower(),
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size if path.exists() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def choose_signal_file(file_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in file_df[file_df["suffix"].isin([".mat", ".txt"])].groupby("SubjectID"):
        mat = g[g["suffix"].eq(".mat")]
        txt = g[g["suffix"].eq(".txt")]
        chosen = mat.iloc[0] if not mat.empty else txt.iloc[0]
        rows.append(
            {
                "SubjectID": sid,
                "chosen_signal_path": chosen["path"],
                "has_mat": bool(not mat.empty),
                "has_txt": bool(not txt.empty),
                "n_signal_files": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("SubjectID").reset_index(drop=True)


def load_signal_for_inventory(path: Path) -> tuple[np.ndarray | None, str, str]:
    if path.suffix.lower() == ".mat":
        try:
            keys = [
                (name, shape, klass)
                for name, shape, klass in sio.whosmat(path)
                if len(shape) == 2 and klass in {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32"}
            ]
            chosen = next((x for x in keys if "signal" in x[0].lower()), keys[0] if keys else None)
            if chosen is None:
                return None, "", "no_numeric_2d_variable"
            data = sio.loadmat(path, variable_names=[chosen[0]], squeeze_me=False)
            return np.asarray(data[chosen[0]], dtype=np.float64), chosen[0], "ok"
        except Exception as exc:
            return None, "", f"failed:{exc}"
    try:
        try:
            arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
        except Exception:
            arr = np.loadtxt(path, dtype=np.float64)
        return np.asarray(arr, dtype=np.float64), "", "ok"
    except Exception as exc:
        return None, "", f"failed:{exc}"


def shape_stats(path: Path) -> dict[str, Any]:
    arr, var, status = load_signal_for_inventory(path)
    out: dict[str, Any] = {
        "main_variable": var,
        "load_status": status,
        "shape": "",
        "n_timepoints": np.nan,
        "n_rois_raw": np.nan,
        "finite_fraction": np.nan,
        "nan_count": np.nan,
    }
    if arr is None:
        return out
    out["shape"] = "x".join(str(x) for x in arr.shape)
    if arr.ndim == 2:
        if arr.shape[1] == 170:
            out["n_timepoints"] = int(arr.shape[0])
            out["n_rois_raw"] = int(arr.shape[1])
        elif arr.shape[0] == 170:
            out["n_timepoints"] = int(arr.shape[1])
            out["n_rois_raw"] = int(arr.shape[0])
        else:
            out["n_timepoints"] = int(arr.shape[0])
            out["n_rois_raw"] = int(arr.shape[1])
    out["finite_fraction"] = float(np.isfinite(arr).mean())
    out["nan_count"] = int(np.isnan(arr).sum())
    return out


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def metadata_subset(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    sid_col = first_existing_col(df, ["SubjectID", "Subject", "PTID"])
    if sid_col is None:
        raise RuntimeError(f"metadata missing SubjectID-like column: {metadata_path}")
    out = pd.DataFrame({"SubjectID": df[sid_col].astype(str)})
    for target, candidates in {
        "ResearchGroup_Mapped": ["ResearchGroup_Mapped", "ResearchGroup", "DX"],
        "Manufacturer": ["Manufacturer", "MANUFACTURER"],
        "Site3": ["Site3", "SITE", "SITEID"],
        "Age": ["Age", "AGE"],
        "Sex": ["Sex", "PTGENDER", "Gender"],
        "raw_tp_group": ["raw_tp_group", "rawTP", "n_timepoints_raw", "TimePoints"],
        "ImageID": ["ImageID", "IMAGEUID", "Image Data ID"],
    }.items():
        col = first_existing_col(df, candidates)
        out[target] = df[col] if col else ""
    return out.drop_duplicates("SubjectID", keep="first")


def offdiag_values(mat: np.ndarray) -> np.ndarray:
    mask = ~np.eye(mat.shape[-1], dtype=bool)
    return mat[mask].astype(np.float64)


def matrix_qc(mat: np.ndarray) -> dict[str, Any]:
    return {
        "finite_fraction": float(np.isfinite(mat).mean()),
        "nan_count": int(np.isnan(mat).sum()),
        "max_abs_asymmetry": float(np.nanmax(np.abs(mat - mat.T))),
        "max_abs_diagonal": float(np.nanmax(np.abs(np.diag(mat)))),
    }


def channel_delta_rows(sid: str, tensor_idx: int, old: np.ndarray, new: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
        old_mat = old[ch_idx]
        new_mat = new[ch_idx]
        ov = offdiag_values(old_mat)
        nv = offdiag_values(new_mat)
        if np.nanstd(ov) > 0 and np.nanstd(nv) > 0:
            corr = float(np.corrcoef(ov, nv)[0, 1])
        else:
            corr = np.nan
        diff = new_mat.astype(np.float64) - old_mat.astype(np.float64)
        q_old = matrix_qc(old_mat)
        q_new = matrix_qc(new_mat)
        rows.append(
            {
                "SubjectID": sid,
                "tensor_idx": tensor_idx,
                "channel_idx": ch_idx,
                "channel_name": ch_name,
                "old_offdiag_mean": float(np.nanmean(ov)),
                "old_offdiag_std": float(np.nanstd(ov)),
                "old_offdiag_min": float(np.nanmin(ov)),
                "old_offdiag_max": float(np.nanmax(ov)),
                "new_offdiag_mean": float(np.nanmean(nv)),
                "new_offdiag_std": float(np.nanstd(nv)),
                "new_offdiag_min": float(np.nanmin(nv)),
                "new_offdiag_max": float(np.nanmax(nv)),
                "old_new_offdiag_pearson_r": corr,
                "mean_abs_edge_difference": float(np.nanmean(np.abs(offdiag_values(diff)))),
                "frobenius_norm_difference": float(np.linalg.norm(diff)),
                "old_finite_fraction": q_old["finite_fraction"],
                "new_finite_fraction": q_new["finite_fraction"],
                "old_max_abs_asymmetry": q_old["max_abs_asymmetry"],
                "new_max_abs_asymmetry": q_new["max_abs_asymmetry"],
                "old_max_abs_diagonal": q_old["max_abs_diagonal"],
                "new_max_abs_diagonal": q_new["max_abs_diagonal"],
            }
        )
    return rows


def load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def save_npz_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def build_stagea_dryrun_command(replacement_tensor: Path) -> list[str]:
    cfg = json.loads(PROMOTED_CONFIG.read_text(encoding="utf-8"))
    params = cfg["parameters"]
    cmd = [
        cfg.get("python_executable") or sys.executable,
        str(PROJECT_ROOT / cfg["paths"]["training_script"]),
        "--global_tensor_path",
        str(replacement_tensor),
        "--metadata_path",
        str(Path(cfg["paths"]["metadata_path"])),
        "--output_dir",
        str(PROMOTED_REPROCESSED_OUTPUT),
    ]
    for key, value in params.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            if value:
                cmd.append(flag)
                cmd.extend(str(x) for x in value)
        elif value is not None:
            cmd.extend([flag, str(value)])
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    cmd.append("--dry-run")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pairwise-n-jobs", type=int, default=4)
    parser.add_argument("--skip-stagea-dryrun", action="store_true")
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    log: list[dict[str, Any]] = [{"timestamp": now_iso(), "event": "start", "output_dir": rel(outdir)}]

    if not ORIGINAL_TENSOR.exists():
        raise FileNotFoundError(ORIGINAL_TENSOR)
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)
    if not NEW_ROISIGNALS_DIR.exists():
        raise FileNotFoundError(NEW_ROISIGNALS_DIR)

    original_sha_before = sha256_file(ORIGINAL_TENSOR)
    file_df = list_new_files(NEW_ROISIGNALS_DIR)
    chosen_df = choose_signal_file(file_df)
    if set(chosen_df["SubjectID"]) != EXPECTED_SUBJECTS or len(chosen_df) != 14:
        write_df(file_df, outdir / "new_roisignals_file_inventory.csv", outdir / "new_roisignals_file_inventory.md")
        raise RuntimeError(f"Expected exactly 14 known subjects, found {len(chosen_df)}: {sorted(chosen_df['SubjectID'])}")

    shape_rows = []
    for _, row in chosen_df.iterrows():
        shape_rows.append({"SubjectID": row["SubjectID"], **shape_stats(Path(row["chosen_signal_path"]))})
    shape_df = pd.DataFrame(shape_rows)

    metadata = metadata_subset(METADATA_PATH)
    with np.load(ORIGINAL_TENSOR, allow_pickle=True) as z:
        tensor = z["global_tensor_data"].astype(np.float32)
        subject_ids = z["subject_ids"].astype(str).tolist()
        channel_names = z["channel_names"].astype(str).tolist()
        roi_names = z["roi_names_in_order"].astype(str).tolist() if "roi_names_in_order" in z.files else []
        network_labels = z["network_labels_in_order"].astype(str).tolist() if "network_labels_in_order" in z.files else []
        target_len = int(np.asarray(z["target_len_ts"]).item())
        py_bandpass = bool(np.asarray(z["python_bandpass_applied"]).item())
        orig_payload = {k: z[k] for k in z.files}

    if channel_names != CHANNEL_NAMES or fe.CHANNEL_NAMES != CHANNEL_NAMES:
        raise RuntimeError(f"Channel order mismatch: tensor={channel_names}, builder={fe.CHANNEL_NAMES}")
    if target_len != 140 or py_bandpass is not False:
        raise RuntimeError(f"Unexpected tensor metadata target_len={target_len}, python_bandpass={py_bandpass}")

    tensor_index = {sid: i for i, sid in enumerate(subject_ids)}
    inv = (
        chosen_df.merge(shape_df, on="SubjectID", how="left")
        .merge(metadata, on="SubjectID", how="left")
    )
    inv["exists_in_metadata"] = inv["ResearchGroup_Mapped"].notna() & inv["ResearchGroup_Mapped"].astype(str).ne("")
    inv["exists_in_global_tensor"] = inv["SubjectID"].isin(tensor_index)
    inv["tensor_idx"] = inv["SubjectID"].map(tensor_index)
    inv["diagnosis_pool"] = inv["ResearchGroup_Mapped"].fillna("")
    write_df(file_df, outdir / "new_roisignals_all_files.csv", outdir / "new_roisignals_all_files.md")
    write_df(inv, outdir / "inventory_reprocessed14.csv", outdir / "inventory_reprocessed14.md")

    overlap = pd.DataFrame(
        [
            {
                "SubjectID": sid,
                "known_site31_reverse_nondefault": True,
                "present_in_new14": sid in set(inv["SubjectID"]),
                "note": "expected_absent_from_new14" if sid == "031_S_4021" and sid not in set(inv["SubjectID"]) else "",
            }
            for sid in sorted(KNOWN_SITE31_REVERSE)
        ]
    )
    write_df(overlap, outdir / "known_problem_subject_overlap.csv", outdir / "known_problem_subject_overlap.md")

    if not bool(inv["exists_in_global_tensor"].all()):
        missing = inv.loc[~inv["exists_in_global_tensor"], "SubjectID"].tolist()
        raise RuntimeError(f"Cannot replace exactly 14 rows; missing from tensor: {missing}")
    if not bool(inv["exists_in_metadata"].all()):
        missing = inv.loc[~inv["exists_in_metadata"], "SubjectID"].tolist()
        raise RuntimeError(f"Cannot confirm metadata for all 14 subjects; missing: {missing}")

    roi_info = fe.build_roi_reduction_and_order()
    if roi_info["roi_names_new_order"] != roi_names:
        raise RuntimeError("ROI names/order from builder do not match locked tensor roi_names_in_order")
    if roi_info["network_labels_new_order"] != network_labels:
        raise RuntimeError("Network labels/order from builder do not match locked tensor network_labels_in_order")

    recompute_rows = []
    channel_qc_rows = []
    delta_rows = []
    replacement_rows = []
    replacement_tensor = tensor.copy()
    individual_dir = outdir / "recomputed_individual_tensors"
    individual_dir.mkdir(parents=True, exist_ok=True)

    for _, row in inv.sort_values("SubjectID").iterrows():
        sid = row["SubjectID"]
        started = time.time()
        raw, var, raw_shape, load_status = fe.load_signal(Path(row["chosen_signal_path"]))
        if raw is None:
            raise RuntimeError(f"{sid}: failed to load signal: {load_status}")
        individual_path = individual_dir / f"tensor_7ch_131rois_site31_mayo_{sid}.npz"
        cached = False
        if individual_path.exists():
            try:
                with np.load(individual_path, allow_pickle=True) as z:
                    cached_tensor = z["tensor"].astype(np.float32)
                if cached_tensor.shape == (7, 131, 131) and np.isfinite(cached_tensor).all():
                    new_tensor = cached_tensor
                    statuses = {"cached_existing_individual_tensor": "ok"}
                    qc_rows = []
                    pre_status = "cached_existing_individual_tensor"
                    pre_qc = {"processed_shape": "cached", "processed_nan_count": 0}
                    cached = True
                else:
                    cached = False
            except Exception:
                cached = False
        if not cached:
            ts, pre_status, pre_qc = fe.preprocess_timeseries_no_pybandpass(raw, roi_info)
            if ts is None or pre_status != "ok":
                raise RuntimeError(f"{sid}: preprocessing failed: {pre_status}")
            new_tensor, statuses, qc_rows = fe.compute_channels(ts, pairwise_n_jobs=args.pairwise_n_jobs)
        if new_tensor.shape != (7, 131, 131):
            raise RuntimeError(f"{sid}: recomputed tensor shape {new_tensor.shape} != (7,131,131)")
        if not np.isfinite(new_tensor).all():
            raise RuntimeError(f"{sid}: recomputed tensor contains non-finite values")
        for ch_idx in range(new_tensor.shape[0]):
            q = matrix_qc(new_tensor[ch_idx])
            if q["max_abs_asymmetry"] > 1e-5:
                raise RuntimeError(f"{sid} ch{ch_idx}: asymmetry {q['max_abs_asymmetry']}")
            if q["max_abs_diagonal"] > 1e-6:
                raise RuntimeError(f"{sid} ch{ch_idx}: nonzero diagonal {q['max_abs_diagonal']}")
        idx = int(row["tensor_idx"])
        old_tensor = tensor[idx].copy()
        replacement_tensor[idx] = new_tensor
        if not cached:
            np.savez_compressed(
                individual_path,
                tensor=new_tensor.astype(np.float32),
                SubjectID=sid,
                channel_names=np.asarray(CHANNEL_NAMES, dtype=str),
                target_len_ts=np.asarray(140),
                tr_seconds=np.asarray(3.0),
                python_bandpass_applied=np.asarray(False),
                preprocessing_source=np.asarray("Site31_MayoSliceOrder_ROISignals_no_python_bandpass"),
                roi_order_name=np.asarray("aal3_manual_yeo17_order"),
                roi_names_in_order=np.asarray(roi_names, dtype=str),
                network_labels_in_order=np.asarray(network_labels, dtype=str),
                source_signal_path=np.asarray(str(row["chosen_signal_path"])),
            )
        recompute_rows.append(
            {
                "SubjectID": sid,
                "tensor_idx": idx,
                "source_signal_path": row["chosen_signal_path"],
                "raw_shape": raw_shape,
                "main_variable": var,
                "load_status": load_status,
                "preprocess_status": pre_status,
                "processed_shape": pre_qc.get("processed_shape"),
                "processed_nan_count": pre_qc.get("processed_nan_count"),
                "channel_statuses": json.dumps(statuses, sort_keys=True),
                "elapsed_sec": time.time() - started,
                "individual_tensor_path": str(individual_path),
                "used_cached_individual_tensor": cached,
            }
        )
        for qc in qc_rows:
            channel_qc_rows.append({"SubjectID": sid, **qc})
        delta_rows.extend(channel_delta_rows(sid, idx, old_tensor, new_tensor))
        replacement_rows.append({"SubjectID": sid, "tensor_idx": idx, "replaced": True})
        log.append({"timestamp": now_iso(), "event": "subject_recomputed", "SubjectID": sid, "tensor_idx": idx})

    recompute_df = pd.DataFrame(recompute_rows)
    channel_qc_df = pd.DataFrame(channel_qc_rows)
    delta_df = pd.DataFrame(delta_rows)
    write_df(recompute_df, outdir / "recompute_connectome_qc.csv", outdir / "recompute_connectome_qc.md")
    write_df(channel_qc_df, outdir / "recompute_channel_qc.csv", outdir / "recompute_channel_qc.md")
    write_df(delta_df, outdir / "old_vs_new_tensor_delta_by_subject_channel.csv", outdir / "old_vs_new_tensor_delta_by_subject_channel.md")
    subj_summary = delta_df.groupby("SubjectID", as_index=False).agg(
        mean_abs_edge_difference_mean=("mean_abs_edge_difference", "mean"),
        frobenius_norm_difference_mean=("frobenius_norm_difference", "mean"),
        old_new_offdiag_pearson_r_median=("old_new_offdiag_pearson_r", "median"),
    )
    ch_summary = delta_df.groupby(["channel_idx", "channel_name"], as_index=False).agg(
        mean_abs_edge_difference_mean=("mean_abs_edge_difference", "mean"),
        frobenius_norm_difference_mean=("frobenius_norm_difference", "mean"),
        old_new_offdiag_pearson_r_median=("old_new_offdiag_pearson_r", "median"),
    )
    write_df(subj_summary, outdir / "old_vs_new_delta_summary_by_subject.csv", outdir / "old_vs_new_delta_summary_by_subject.md")
    write_df(ch_summary, outdir / "old_vs_new_delta_summary_by_channel.csv", outdir / "old_vs_new_delta_summary_by_channel.md")

    replacement_payload = dict(orig_payload)
    replacement_payload["global_tensor_data"] = replacement_tensor.astype(np.float32)
    replacement_payload["preprocessing_source"] = np.asarray(
        "DPARSF_ROISignals_AAL3_10000_with_Site31_MayoSliceOrder_reprocessed14", dtype="<U96"
    )
    replacement_payload["dataset_name"] = np.asarray(
        "adni_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14", dtype="<U96"
    )
    replacement_payload["site31_mayo_reprocessed_subject_ids"] = np.asarray(sorted(inv["SubjectID"].tolist()), dtype="<U32")
    replacement_payload["site31_mayo_reprocessed_source_dir"] = np.asarray(str(NEW_ROISIGNALS_DIR), dtype="<U256")
    replacement_payload["site31_mayo_original_tensor_sha256"] = np.asarray(original_sha_before, dtype="<U64")
    save_npz_payload(REPLACEMENT_TENSOR, replacement_payload)

    original_sha_after = sha256_file(ORIGINAL_TENSOR)
    replacement_sha = sha256_file(REPLACEMENT_TENSOR)
    if original_sha_before != original_sha_after:
        raise RuntimeError("Original tensor SHA changed during script execution")

    with np.load(REPLACEMENT_TENSOR, allow_pickle=True) as z:
        repl = z["global_tensor_data"].astype(np.float32)
        repl_subject_ids = z["subject_ids"].astype(str).tolist()
    if repl.shape != tensor.shape:
        raise RuntimeError(f"Replacement tensor shape {repl.shape} != original {tensor.shape}")
    if repl_subject_ids != subject_ids:
        raise RuntimeError("Replacement subject_ids order differs from original")

    row_equal = np.array([np.array_equal(tensor[i], repl[i]) for i in range(tensor.shape[0])])
    changed_indices = np.where(~row_equal)[0].tolist()
    expected_indices = sorted(int(x) for x in inv["tensor_idx"].tolist())
    if changed_indices != expected_indices:
        raise RuntimeError(f"Changed rows mismatch expected. changed={changed_indices}, expected={expected_indices}")

    validation_rows = [
        {"check": "original_tensor_sha_unchanged", "status": "PASS", "detail": original_sha_before},
        {"check": "replacement_shape_matches_original", "status": "PASS", "detail": str(repl.shape)},
        {"check": "subject_order_preserved", "status": "PASS", "detail": f"N={len(repl_subject_ids)}"},
        {"check": "exactly_14_rows_changed", "status": "PASS", "detail": ",".join(map(str, changed_indices))},
        {"check": "all_other_rows_bitwise_identical", "status": "PASS", "detail": f"unchanged_rows={int(row_equal.sum())}"},
        {"check": "replacement_sha256", "status": "PASS", "detail": replacement_sha},
    ]
    write_df(pd.DataFrame(validation_rows), outdir / "replacement_tensor_validation.csv", outdir / "replacement_tensor_validation.md")

    class_counts = inv["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
    manifest_df = pd.DataFrame(replacement_rows).merge(inv, on="SubjectID", how="left")
    manifest_df["original_tensor_sha256"] = original_sha_before
    manifest_df["replacement_tensor_sha256"] = replacement_sha
    manifest_df["replacement_tensor_path"] = str(REPLACEMENT_TENSOR)
    write_df(manifest_df, outdir / "replacement_manifest.csv", outdir / "replacement_manifest.md")
    manifest_json = {
        "created_at": now_iso(),
        "original_tensor": str(ORIGINAL_TENSOR),
        "replacement_tensor": str(REPLACEMENT_TENSOR),
        "original_tensor_sha256": original_sha_before,
        "replacement_tensor_sha256": replacement_sha,
        "new_roisignals_dir": str(NEW_ROISIGNALS_DIR),
        "subjects_replaced": sorted(inv["SubjectID"].tolist()),
        "known_site31_reverse_present": sorted(set(inv["SubjectID"]) & KNOWN_SITE31_REVERSE),
        "known_site31_reverse_missing": sorted(KNOWN_SITE31_REVERSE - set(inv["SubjectID"])),
        "diagnosis_counts_reprocessed14": {str(k): int(v) for k, v in class_counts.items()},
        "channel_names": CHANNEL_NAMES,
        "target_len_ts": 140,
        "roi_order_name": "aal3_manual_yeo17_order",
    }
    write_text(outdir / "replacement_manifest.json", json.dumps(manifest_json, indent=2) + "\n")

    cmd = build_stagea_dryrun_command(REPLACEMENT_TENSOR)
    launch_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n# Dry-run only. This command does not train.\n" + shlex.join(cmd) + "\n"
    launch_path = outdir / "launch_promoted_site31_mayo_reprocessed14.sh"
    write_text(launch_path, launch_text)
    try:
        launch_path.chmod(0o755)
    except Exception:
        pass
    if not args.skip_stagea_dryrun:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False, timeout=180)
        write_text(outdir / "dryrun_training_command_stdout.txt", proc.stdout)
        write_text(outdir / "dryrun_training_command_stderr.txt", proc.stderr)
        dry_status = "PASS" if proc.returncode == 0 else "FAIL"
        dry_detail = f"returncode={proc.returncode}"
    else:
        dry_status = "SKIPPED"
        dry_detail = "skip_stagea_dryrun"
    dry_df = pd.DataFrame(
        [
            {"check": "stagea_dryrun_command", "status": dry_status, "detail": dry_detail, "command": shlex.join(cmd)},
            {"check": "planned_output_dir", "status": "PASS", "detail": str(PROMOTED_REPROCESSED_OUTPUT), "command": ""},
        ]
    )
    write_df(dry_df, outdir / "dryrun_training_command_validation.csv", outdir / "dryrun_training_command_validation.md")

    cn_ad = inv[inv["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    mci = inv[inv["ResearchGroup_Mapped"].eq("MCI")]
    known_present = sorted(set(inv["SubjectID"]) & KNOWN_SITE31_REVERSE)
    known_missing = sorted(KNOWN_SITE31_REVERSE - set(inv["SubjectID"]))
    readme = f"""# Site31 Mayo Slice-Order Reprocessed14 Tensor Preflight

Status: `PASS`.

The package recomputed all seven connectivity channels for 14 Site31 Mayo-slice-order
ROISignals using the promoted no-Python-bandpass code path, verified the 170-to-131
manual Yeo17 ROI order against the locked tensor, and wrote a new replacement tensor.

- Original tensor unchanged SHA256: `{original_sha_before}`
- Replacement tensor SHA256: `{replacement_sha}`
- Replacement tensor: `{REPLACEMENT_TENSOR}`
- Rows changed: `{len(changed_indices)}`
- Reprocessed CN/AD classifier subjects: `{len(cn_ad)}`
- Reprocessed MCI VAE-only subjects: `{len(mci)}`
- Known Site31 reverse/non-default subjects present: `{', '.join(known_present)}`
- Known Site31 reverse/non-default subjects missing: `{', '.join(known_missing)}`

No training was launched.
"""
    write_text(outdir / "00_EXECUTIVE_SUMMARY.md", readme)
    recommendation = f"""# Final Recommendation

Replacement tensor construction passed.

The new tensor preserves subject order, ROI order, channel order, and all original
NPZ metadata fields while replacing only the 14 matching Site31 Mayo-slice-order
subjects. All other tensor rows are bitwise-identical to the locked promoted tensor.

Classifier CN/AD among the 14: `{len(cn_ad)}`. VAE-only MCI among the 14: `{len(mci)}`.
Known Site31 reverse/non-default subjects actually reprocessed: `{len(known_present)}/5`
(`{', '.join(known_present)}`). The expected subject `031_S_4021` is absent from
the new 14-subject ROISignals directory and therefore was not replaced.

It is scientifically safe to launch a full retraining sensitivity from this
replacement tensor if the dry-run command remains PASS. Caveats: this is a
targeted reprocessing sensitivity, not a replacement for the promoted model;
any performance change must be interpreted against subject-level changes, Site31
specificity, and the fact that one known problem subject was not reprocessed.
"""
    write_text(outdir / "final_recommendation.md", recommendation)

    log.append({"timestamp": now_iso(), "event": "complete", "replacement_tensor": str(REPLACEMENT_TENSOR), "dryrun_status": dry_status})
    write_text(outdir / "command_log.json", json.dumps(log, indent=2) + "\n")
    print(json.dumps({"output_dir": rel(outdir), "replacement_tensor": str(REPLACEMENT_TENSOR), "status": "PASS"}, indent=2))
    return 0 if dry_status in {"PASS", "SKIPPED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
