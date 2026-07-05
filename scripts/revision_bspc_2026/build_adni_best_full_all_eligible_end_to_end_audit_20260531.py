#!/usr/bin/env python
"""Build a read-only end-to-end audit package for the best all-eligible ADNI model.

This script only reads existing tensors, metadata, and model artifacts. It writes
summary tables, plots, and recommendations under the revision results tree.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results/revision_bspc_2026/adni_best_full_all_eligible_end_to_end_audit_20260531"

ALL_ELIGIBLE_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1c_recover035_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1c_recover035_no_pybandpass.npz"
)
ALL_ELIGIBLE_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1c_recover035_no_pybandpass/"
    "training_ready_metadata_v5_1c_recover035_no_pybandpass.csv"
)
BASE_V51B_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
BASE_V51B_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

MISSING_SUBJECT_DECISIONS = ROOT / "results/revision_bspc_2026/tensor_metadata_missing_subjects_audit/final_action_table.csv"

BEST_RUN_DIR = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
BEST_READOUT_DIR = BEST_RUN_DIR / "classifier_only_readout"
BEST_CALIB_DIR = ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_stageB_oof_score_calibration"
LOGREG_C_GRID_AUDIT = ROOT / "results/revision_bspc_2026/logreg_c_grid_audit_3models/selected_C_by_fold.csv"

SELECTED_CHANNEL_INDICES = [1, 0, 2]
SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, stem: str, out_dir: Path = OUTPUT_DIR, max_md_rows: int | None = 80) -> None:
    ensure_dir(out_dir)
    csv_path = out_dir / f"{stem}.csv"
    md_path = out_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_df = df if max_md_rows is None else df.head(max_md_rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {stem}\n\n")
        f.write(f"Rows: {len(df)}\n\n")
        if len(df) > len(md_df):
            f.write(f"Showing first {len(md_df)} rows in Markdown; full table is in CSV.\n\n")
        try:
            f.write(md_df.to_markdown(index=False))
        except Exception:
            f.write(md_df.to_csv(index=False))
        f.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    text = str(value)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


def normalize_subject_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_tensor(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def offdiag_values(mat: np.ndarray, tri: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return mat[..., tri[0], tri[1]]


def summarize_numeric(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def best_epoch_from_history(history: dict[str, Any]) -> int:
    key = "val_loss_modelsel" if "val_loss_modelsel" in history else "val_loss"
    values = np.asarray(history.get(key, []), dtype=float)
    if values.size == 0:
        return -1
    return int(np.nanargmin(values) + 1)


def slope_last(values: Iterable[float], n: int) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    tail = arr[-min(n, arr.size) :]
    if tail.size < 2:
        return float("nan")
    x = np.arange(tail.size, dtype=float)
    return float(np.polyfit(x, tail, deg=1)[0])


def get_at_epoch(seq: Any, epoch_1based: int) -> float:
    try:
        arr = np.asarray(seq, dtype=float)
        if epoch_1based < 1 or epoch_1based > arr.size:
            return float("nan")
        return float(arr[epoch_1based - 1])
    except Exception:
        return float("nan")


def age_bin(age: Any) -> str:
    a = safe_float(age)
    if not np.isfinite(a):
        return "missing"
    if a < 65:
        return "<65"
    if a < 75:
        return "65-74"
    if a < 85:
        return "75-84"
    return "85+"


def build_subject_inclusion_ledger(
    tensor: dict[str, Any],
    metadata: pd.DataFrame,
    base_tensor: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tensor_subjects = [normalize_subject_id(x) for x in tensor["subject_ids"]]
    tensor_set = set(tensor_subjects)
    metadata = metadata.copy()
    metadata["SubjectID"] = metadata["SubjectID"].map(normalize_subject_id)
    md_set = set(metadata["SubjectID"])

    base_subjects: list[str] = []
    if base_tensor is not None:
        base_subjects = [normalize_subject_id(x) for x in base_tensor["subject_ids"]]
    base_set = set(base_subjects)

    decision_df = read_csv_if_exists(MISSING_SUBJECT_DECISIONS)
    decisions = {}
    if not decision_df.empty:
        for _, row in decision_df.iterrows():
            decisions[normalize_subject_id(row.get("SubjectID"))] = row.to_dict()

    rows: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        sid = normalize_subject_id(row["SubjectID"])
        dx = row.get("ResearchGroup_Mapped")
        has_dx = str(dx) in {"CN", "MCI", "AD"}
        has_age_sex = pd.notna(row.get("Age")) and pd.notna(row.get("Sex")) and str(row.get("Sex")).strip() != ""
        in_tensor = sid in tensor_set
        training_ready = bool(row.get("training_ready", True))
        reason = "included_qc_valid_metadata_valid"
        status = "included"
        if not in_tensor:
            reason = "missing_tensor"
            status = "excluded"
        elif not has_dx:
            reason = "missing_diagnosis"
            status = "excluded"
        elif not has_age_sex:
            reason = "missing_age_or_sex"
            status = "excluded"
        elif not training_ready:
            reason = "failed_qc_or_not_training_ready"
            status = "excluded"
        rows.append(
            {
                "SubjectID": sid,
                "tensor_index": row.get("tensor_index"),
                "ResearchGroup_Mapped": dx,
                "Age": row.get("Age"),
                "Sex": row.get("Sex"),
                "Manufacturer": row.get("Manufacturer"),
                "Site3": row.get("Site3"),
                "ImageID": row.get("ImageID"),
                "Visit": row.get("Visit"),
                "in_base_v5_1b_tensor": sid in base_set,
                "in_all_eligible_tensor": in_tensor,
                "in_all_eligible_metadata": sid in md_set,
                "training_ready": training_ready,
                "in_vae_pool": status == "included" and has_dx,
                "in_classifier_pool": status == "included" and str(dx) in {"CN", "AD"},
                "inclusion_status": status,
                "reason_code": reason,
                "reason_detail": "",
                "current_action": "include in all-eligible audit/model pool" if status == "included" else "exclude",
                "future_action": "",
            }
        )

    # Add base tensor-only subjects that are intentionally not in the all-eligible clean branch.
    for sid in sorted(base_set - tensor_set):
        dec = decisions.get(sid, {})
        rows.append(
            {
                "SubjectID": sid,
                "tensor_index": dec.get("tensor_index", np.nan),
                "ResearchGroup_Mapped": dec.get("diagnosis_status", "unknown"),
                "Age": np.nan,
                "Sex": np.nan,
                "Manufacturer": np.nan,
                "Site3": sid[:3] if sid else "",
                "ImageID": np.nan,
                "Visit": np.nan,
                "in_base_v5_1b_tensor": True,
                "in_all_eligible_tensor": False,
                "in_all_eligible_metadata": False,
                "training_ready": False,
                "in_vae_pool": False,
                "in_classifier_pool": False,
                "inclusion_status": "excluded",
                "reason_code": "corrupted_no_image_or_unresolved_metadata_qc",
                "reason_detail": dec.get("signal_status", ""),
                "current_action": dec.get("current_action", "exclude"),
                "future_action": dec.get("future_action", ""),
            }
        )

    # Add explicit decision rows if absent from both base and all-eligible sets.
    for sid, dec in decisions.items():
        if sid and sid not in {r["SubjectID"] for r in rows}:
            rows.append(
                {
                    "SubjectID": sid,
                    "tensor_index": dec.get("tensor_index", np.nan),
                    "ResearchGroup_Mapped": dec.get("diagnosis_status", "unknown"),
                    "Age": np.nan,
                    "Sex": np.nan,
                    "Manufacturer": np.nan,
                    "Site3": sid[:3],
                    "ImageID": np.nan,
                    "Visit": np.nan,
                    "in_base_v5_1b_tensor": sid in base_set,
                    "in_all_eligible_tensor": sid in tensor_set,
                    "in_all_eligible_metadata": sid in md_set,
                    "training_ready": False,
                    "in_vae_pool": False,
                    "in_classifier_pool": False,
                    "inclusion_status": "excluded",
                    "reason_code": "documented_exclusion",
                    "reason_detail": dec.get("root_cause", ""),
                    "current_action": dec.get("current_action", "exclude"),
                    "future_action": dec.get("future_action", ""),
                }
            )

    ledger = pd.DataFrame(rows).sort_values(["inclusion_status", "SubjectID"], ascending=[True, True])
    duplicate_subjects = metadata["SubjectID"][metadata["SubjectID"].duplicated()].tolist()
    summary = {
        "all_eligible_tensor_subjects": len(tensor_subjects),
        "all_eligible_metadata_subjects": int(len(metadata)),
        "matched_tensor_metadata_subjects": int(len(tensor_set & md_set)),
        "base_v5_1b_tensor_subjects": len(base_subjects),
        "vae_pool_counts": metadata.loc[metadata["ResearchGroup_Mapped"].isin(["CN", "MCI", "AD"]), "ResearchGroup_Mapped"]
        .value_counts()
        .to_dict(),
        "classifier_pool_counts": metadata.loc[metadata["ResearchGroup_Mapped"].isin(["CN", "AD"]), "ResearchGroup_Mapped"]
        .value_counts()
        .to_dict(),
        "excluded_subjects_in_ledger": int((ledger["inclusion_status"] == "excluded").sum()),
        "duplicate_subjects": duplicate_subjects,
    }
    return ledger, summary


def build_tensor_qc(tensor: dict[str, Any], metadata: pd.DataFrame) -> pd.DataFrame:
    x = tensor["global_tensor_data"]
    subjects = [normalize_subject_id(s) for s in tensor["subject_ids"]]
    channels = [str(c) for c in tensor["channel_names"]]
    tri = np.triu_indices(x.shape[-1], k=1)
    rows: list[dict[str, Any]] = []
    for ch_idx in SELECTED_CHANNEL_INDICES:
        ch_name = channels[ch_idx]
        data = x[:, ch_idx]
        off = offdiag_values(data, tri)
        diag = np.diagonal(data, axis1=1, axis2=2)
        sym_abs = np.abs(data - np.swapaxes(data, 1, 2))
        stats = summarize_numeric(off.reshape(-1))
        rows.append(
            {
                "tensor_path": str(ALL_ELIGIBLE_TENSOR),
                "tensor_sha256": sha256_file(ALL_ELIGIBLE_TENSOR),
                "n_subjects": x.shape[0],
                "n_channels_available": x.shape[1],
                "n_roi": x.shape[2],
                "selected_channel_index": ch_idx,
                "selected_channel_name": ch_name,
                "expected_selected_channel_name": SELECTED_CHANNEL_NAMES[SELECTED_CHANNEL_INDICES.index(ch_idx)],
                "nan_count": int(np.isnan(data).sum()),
                "inf_count": int(np.isinf(data).sum()),
                "diag_abs_max": float(np.nanmax(np.abs(diag))),
                "diag_abs_mean": float(np.nanmean(np.abs(diag))),
                "sym_abs_max": float(np.nanmax(sym_abs)),
                "sym_abs_mean": float(np.nanmean(sym_abs)),
                **{f"offdiag_{k}": v for k, v in stats.items()},
            }
        )

    # Add global tensor/channel metadata as pseudo rows for audit clarity.
    return pd.DataFrame(rows)


def build_channel_group_summary(tensor: dict[str, Any], metadata: pd.DataFrame) -> pd.DataFrame:
    x = tensor["global_tensor_data"]
    subjects = [normalize_subject_id(s) for s in tensor["subject_ids"]]
    md = metadata.copy()
    md["SubjectID"] = md["SubjectID"].map(normalize_subject_id)
    md = md.set_index("SubjectID").reindex(subjects).reset_index()
    tri = np.triu_indices(x.shape[-1], k=1)
    channels = [str(c) for c in tensor["channel_names"]]
    rows = []
    for ch_idx in SELECTED_CHANNEL_INDICES:
        subject_means = np.nanmean(offdiag_values(x[:, ch_idx], tri), axis=1)
        temp = md.copy()
        temp["subject_offdiag_mean"] = subject_means
        for group_col in ["ResearchGroup_Mapped", "Manufacturer"]:
            for group_value, sub in temp.groupby(group_col, dropna=False):
                rows.append(
                    {
                        "selected_channel_index": ch_idx,
                        "selected_channel_name": channels[ch_idx],
                        "group_col": group_col,
                        "group_value": group_value,
                        "n": int(len(sub)),
                        **{f"subject_offdiag_mean_{k}": v for k, v in summarize_numeric(sub["subject_offdiag_mean"].to_numpy()).items()},
                    }
                )
    return pd.DataFrame(rows)


def plot_tensor_figures(tensor: dict[str, Any], metadata: pd.DataFrame) -> pd.DataFrame:
    fig_dir = OUTPUT_DIR / "channel_distribution_plots"
    ensure_dir(fig_dir)
    x = tensor["global_tensor_data"]
    subjects = [normalize_subject_id(s) for s in tensor["subject_ids"]]
    md = metadata.copy()
    md["SubjectID"] = md["SubjectID"].map(normalize_subject_id)
    md = md.set_index("SubjectID").reindex(subjects).reset_index()
    channels = [str(c) for c in tensor["channel_names"]]
    roi_names = [str(r) for r in tensor.get("roi_names_in_order", np.arange(x.shape[-1]))]
    network_labels = [str(n) for n in tensor.get("network_labels_in_order", ["unknown"] * x.shape[-1])]
    tri = np.triu_indices(x.shape[-1], k=1)

    rows = []
    rng = np.random.default_rng(42)
    for ch_idx in SELECTED_CHANNEL_INDICES:
        ch_name = channels[ch_idx]
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", ch_name)
        off = offdiag_values(x[:, ch_idx], tri).reshape(-1)
        off = off[np.isfinite(off)]
        if off.size > 750_000:
            off_plot = rng.choice(off, size=750_000, replace=False)
        else:
            off_plot = off

        plt.figure(figsize=(8, 5))
        plt.hist(off_plot, bins=120, color="#335f8a", alpha=0.85)
        plt.title(f"Off-diagonal distribution: {ch_name}")
        plt.xlabel("Connectivity value")
        plt.ylabel("Count")
        plt.tight_layout()
        hist_path = fig_dir / f"hist_{ch_idx}_{safe_name}.png"
        plt.savefig(hist_path, dpi=160)
        plt.close()
        rows.append({"figure_type": "histogram", "channel": ch_name, "path": rel(hist_path)})

        cn_idx = md.index[md["ResearchGroup_Mapped"].eq("CN")].to_numpy()
        ad_idx = md.index[md["ResearchGroup_Mapped"].eq("AD")].to_numpy()
        mean_cn = np.nanmean(x[cn_idx, ch_idx], axis=0)
        mean_ad = np.nanmean(x[ad_idx, ch_idx], axis=0)
        diff = mean_ad - mean_cn
        vmax = float(np.nanpercentile(np.abs(np.r_[mean_cn.reshape(-1), mean_ad.reshape(-1)]), 99))
        diff_vmax = float(np.nanpercentile(np.abs(diff.reshape(-1)), 99))
        for label, mat, lim in [
            ("mean_CN", mean_cn, vmax),
            ("mean_AD", mean_ad, vmax),
            ("AD_minus_CN", diff, diff_vmax),
        ]:
            plt.figure(figsize=(6, 5))
            plt.imshow(mat, cmap="coolwarm", vmin=-lim, vmax=lim)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"{label}: {ch_name}")
            plt.tight_layout()
            path = fig_dir / f"{label}_{ch_idx}_{safe_name}.png"
            plt.savefig(path, dpi=160)
            plt.close()
            rows.append({"figure_type": label, "channel": ch_name, "path": rel(path)})

    # Network-pair summaries.
    net_order = list(dict.fromkeys(network_labels))
    net_idx = {net: np.where(np.asarray(network_labels) == net)[0] for net in net_order}
    net_rows = []
    for ch_idx in SELECTED_CHANNEL_INDICES:
        ch_name = channels[ch_idx]
        cn_idx = md.index[md["ResearchGroup_Mapped"].eq("CN")].to_numpy()
        ad_idx = md.index[md["ResearchGroup_Mapped"].eq("AD")].to_numpy()
        mean_cn = np.nanmean(x[cn_idx, ch_idx], axis=0)
        mean_ad = np.nanmean(x[ad_idx, ch_idx], axis=0)
        for a in net_order:
            ia = net_idx[a]
            for b in net_order:
                ib = net_idx[b]
                sub_cn = mean_cn[np.ix_(ia, ib)]
                sub_ad = mean_ad[np.ix_(ia, ib)]
                if a == b:
                    mask = ~np.eye(len(ia), dtype=bool)
                    vals_cn = sub_cn[mask] if mask.size else sub_cn.reshape(-1)
                    vals_ad = sub_ad[mask] if mask.size else sub_ad.reshape(-1)
                else:
                    vals_cn = sub_cn.reshape(-1)
                    vals_ad = sub_ad.reshape(-1)
                net_rows.append(
                    {
                        "channel_index": ch_idx,
                        "channel_name": ch_name,
                        "network_a": a,
                        "network_b": b,
                        "n_roi_a": int(len(ia)),
                        "n_roi_b": int(len(ib)),
                        "mean_cn": float(np.nanmean(vals_cn)) if vals_cn.size else np.nan,
                        "mean_ad": float(np.nanmean(vals_ad)) if vals_ad.size else np.nan,
                        "ad_minus_cn": float(np.nanmean(vals_ad) - np.nanmean(vals_cn)) if vals_cn.size and vals_ad.size else np.nan,
                    }
                )
    write_table(pd.DataFrame(net_rows), "yeo17_network_pair_summary", OUTPUT_DIR, max_md_rows=100)
    write_table(
        pd.DataFrame(
            {
                "roi_index_0based": list(range(len(roi_names))),
                "roi_index_1based": list(range(1, len(roi_names) + 1)),
                "roi_name": roi_names,
                "network_label": network_labels,
            }
        ),
        "roi_order_aal3_yeo17_mapping",
        OUTPUT_DIR,
        max_md_rows=140,
    )
    return pd.DataFrame(rows)


def build_split_audit(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    md = metadata.copy()
    md["SubjectID"] = md["SubjectID"].map(normalize_subject_id)
    md_by_sid = md.set_index("SubjectID")
    md_by_idx = md.set_index("tensor_index")
    for fold in range(1, 6):
        fold_dir = BEST_RUN_DIR / f"fold_{fold}"
        train = read_csv_if_exists(fold_dir / "train_dev_subjects_fold.csv")
        test = read_csv_if_exists(fold_dir / "test_subjects_fold.csv")
        if train.empty or test.empty:
            continue
        train["SubjectID"] = train["SubjectID"].map(normalize_subject_id)
        test["SubjectID"] = test["SubjectID"].map(normalize_subject_id)
        train_set = set(train["SubjectID"])
        test_set = set(test["SubjectID"])
        overlap = sorted(train_set & test_set)
        test_tensor = np.load(fold_dir / "test_tensor_idx.npy") if (fold_dir / "test_tensor_idx.npy").exists() else np.array([])
        vae_pool = np.load(fold_dir / "vae_training_pool_tensor_idx.npy") if (fold_dir / "vae_training_pool_tensor_idx.npy").exists() else np.array([])
        vae_test_overlap = sorted(set(map(int, vae_pool.tolist())) & set(map(int, test_tensor.tolist())))

        for split_name, split_df in [("classifier_train_dev", train), ("classifier_test", test)]:
            merged = split_df[["SubjectID", "ResearchGroup_Mapped", "tensor_idx"]].merge(
                md[["SubjectID", "Manufacturer", "Sex", "Age", "Site3"]],
                on="SubjectID",
                how="left",
            )
            for dx, sub in merged.groupby("ResearchGroup_Mapped", dropna=False):
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "grouping": "ResearchGroup_Mapped",
                        "group": dx,
                        "n": int(len(sub)),
                        "train_test_subject_overlap_n": len(overlap),
                        "train_test_subject_overlap": ",".join(overlap),
                        "vae_pool_outer_test_tensor_overlap_n": len(vae_test_overlap),
                        "vae_pool_outer_test_tensor_overlap": ",".join(map(str, vae_test_overlap)),
                    }
                )
            for col in ["Manufacturer", "Sex"]:
                for val, sub in merged.groupby(col, dropna=False):
                    rows.append(
                        {
                            "fold": fold,
                            "split": split_name,
                            "grouping": col,
                            "group": val,
                            "n": int(len(sub)),
                            "train_test_subject_overlap_n": len(overlap),
                            "train_test_subject_overlap": ",".join(overlap),
                            "vae_pool_outer_test_tensor_overlap_n": len(vae_test_overlap),
                            "vae_pool_outer_test_tensor_overlap": ",".join(map(str, vae_test_overlap)),
                        }
                    )
            merged["AgeBin"] = merged["Age"].map(age_bin)
            for val, sub in merged.groupby("AgeBin", dropna=False):
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "grouping": "AgeBin",
                        "group": val,
                        "n": int(len(sub)),
                        "train_test_subject_overlap_n": len(overlap),
                        "train_test_subject_overlap": ",".join(overlap),
                        "vae_pool_outer_test_tensor_overlap_n": len(vae_test_overlap),
                        "vae_pool_outer_test_tensor_overlap": ",".join(map(str, vae_test_overlap)),
                    }
                )

        if vae_pool.size:
            vae_md = md_by_idx.reindex(vae_pool).reset_index()
            for col in ["ResearchGroup_Mapped", "Manufacturer", "Sex"]:
                for val, sub in vae_md.groupby(col, dropna=False):
                    rows.append(
                        {
                            "fold": fold,
                            "split": "vae_train_pool",
                            "grouping": col,
                            "group": val,
                            "n": int(len(sub)),
                            "train_test_subject_overlap_n": len(overlap),
                            "train_test_subject_overlap": ",".join(overlap),
                            "vae_pool_outer_test_tensor_overlap_n": len(vae_test_overlap),
                            "vae_pool_outer_test_tensor_overlap": ",".join(map(str, vae_test_overlap)),
                        }
                    )
    return pd.DataFrame(rows)


def build_vae_training_qc() -> pd.DataFrame:
    rows = []
    curve_dir = OUTPUT_DIR / "vae_training_curves"
    ensure_dir(curve_dir)
    for fold in range(1, 6):
        fold_dir = BEST_RUN_DIR / f"fold_{fold}"
        hist_path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        rd_path = fold_dir / f"fold_{fold}_rate_distortion.csv"
        if not hist_path.exists():
            continue
        history = joblib.load(hist_path)
        rd = read_csv_if_exists(rd_path)
        best_epoch = best_epoch_from_history(history)
        final_epoch = int(len(history.get("val_loss_modelsel", history.get("val_loss", []))))
        best_idx = best_epoch - 1
        val_loss_modelsel = history.get("val_loss_modelsel", history.get("val_loss", []))
        beta_seq = history.get("beta", [])
        row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "early_stop_epoch": final_epoch,
            "epochs_after_best": final_epoch - best_epoch if best_epoch > 0 else np.nan,
            "reached_max_epoch_10000": bool(final_epoch >= 10000),
            "best_epoch_in_last_10_percent": bool(best_epoch >= 0.9 * 10000),
            "best_val_loss_modelsel": get_at_epoch(val_loss_modelsel, best_epoch),
            "last100_val_loss_modelsel_slope": slope_last(val_loss_modelsel, 100),
            "last300_val_loss_modelsel_slope": slope_last(val_loss_modelsel, 300),
            "train_recon_at_best": get_at_epoch(history.get("train_recon", []), best_epoch),
            "val_recon_at_best": get_at_epoch(history.get("val_recon", []), best_epoch),
            "train_kld_at_best": get_at_epoch(history.get("train_kld", []), best_epoch),
            "val_kld_at_best": get_at_epoch(history.get("val_kld", []), best_epoch),
            "train_kld_over_recon_at_best": get_at_epoch(history.get("train_kld_over_recon", []), best_epoch),
            "val_kld_over_recon_at_best": get_at_epoch(history.get("val_kld_over_recon", []), best_epoch),
            "train_beta_kld_over_recon_at_best": get_at_epoch(history.get("train_beta_kld_over_recon", []), best_epoch),
            "val_beta_kld_over_recon_at_best": get_at_epoch(history.get("val_beta_kld_over_recon", []), best_epoch),
            "beta_at_best": get_at_epoch(beta_seq, best_epoch),
            "lr_T0_80_phase_epoch_mod": int(best_epoch % 80) if best_epoch > 0 else np.nan,
            "beta_cycle_phase_epoch_mod": int(best_epoch % 80) if best_epoch > 0 else np.nan,
        }
        if not rd.empty and best_epoch in set(rd["epoch"].astype(int)):
            rd_best = rd.loc[rd["epoch"].astype(int).eq(best_epoch)].iloc[0]
            for col in ["D_train", "R_train_nats", "L_train_betaMax", "D_val", "R_val_nats", "L_val_betaMax"]:
                row[f"rd_{col}_at_best"] = rd_best.get(col, np.nan)
        rows.append(row)

        # Generate compact curves for the audit package.
        epochs = np.arange(1, final_epoch + 1)
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        for key, label in [("train_recon", "train recon"), ("val_recon", "val recon")]:
            if key in history:
                axes[0].plot(epochs, history[key], label=label, linewidth=1)
        axes[0].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
        axes[0].set_ylabel("Recon")
        axes[0].legend(loc="best")
        for key, label in [("train_kld", "train KLD"), ("val_kld", "val KLD")]:
            if key in history:
                axes[1].plot(epochs, history[key], label=label, linewidth=1)
        axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
        axes[1].set_ylabel("KLD")
        axes[1].legend(loc="best")
        if "val_loss_modelsel" in history:
            axes[2].plot(epochs, history["val_loss_modelsel"], label="val modelsel", linewidth=1)
        if "beta" in history:
            beta = np.asarray(history["beta"], dtype=float)
            if beta.size == final_epoch and np.nanmax(beta) > 0:
                beta_scaled = beta / np.nanmax(beta) * np.nanmax(history["val_loss_modelsel"])
                axes[2].plot(epochs, beta_scaled, label="beta scaled", linewidth=0.8, alpha=0.6)
        axes[2].axvline(best_epoch, color="black", linestyle="--", linewidth=1)
        axes[2].set_ylabel("Val L beta max")
        axes[2].set_xlabel("Epoch")
        axes[2].legend(loc="best")
        fig.suptitle(f"Fold {fold} VAE training curves")
        fig.tight_layout()
        fig.savefig(curve_dir / f"fold_{fold}_vae_training_curves.png", dpi=150)
        plt.close(fig)
    return pd.DataFrame(rows)


def build_latent_qc() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        fold_dir = BEST_RUN_DIR / f"fold_{fold}"
        lq = read_csv_if_exists(fold_dir / "latent_qc_metrics.csv")
        if not lq.empty:
            for _, row in lq.iterrows():
                d = row.to_dict()
                d["source_file"] = rel(fold_dir / "latent_qc_metrics.csv")
                d["summary_type"] = "latent_qc_metrics"
                rows.append(d)
        for split in ["trainDev", "test"]:
            info_path = fold_dir / f"fold_{fold}_{split}_latent_info_summary.csv"
            info = read_csv_if_exists(info_path)
            for _, row in info.iterrows():
                d = {"fold": fold, "split": split, "summary_type": "latent_information", "source_file": rel(info_path)}
                d.update(row.to_dict())
                rows.append(d)
        leak_path = fold_dir / f"fold_{fold}_scanner_leakage_summary.csv"
        leak = read_csv_if_exists(leak_path)
        for _, row in leak.iterrows():
            d = {"fold": fold, "summary_type": "scanner_leakage_train_dev", "source_file": rel(leak_path)}
            d.update(row.to_dict())
            rows.append(d)
    return pd.DataFrame(rows)


def load_classifier_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = read_csv_if_exists(BEST_CALIB_DIR / "calib_pooled_metrics.csv")
    foldwise = read_csv_if_exists(BEST_CALIB_DIR / "calib_foldwise_metrics.csv")
    score_ranges = read_csv_if_exists(BEST_CALIB_DIR / "calib_score_range_by_fold.csv")
    keep_models = ["logreg_l2_original"]
    keep_features = ["z_plus_age_sex"]
    if not pooled.empty:
        pooled = pooled[
            pooled["model_name"].isin(keep_models)
            & pooled["feature_set"].isin(keep_features)
            & pooled["calib_method"].isin(["raw", "oof_logitz", "oof_ecdf"])
        ].copy()
    if not foldwise.empty:
        foldwise = foldwise[
            foldwise["model_name"].isin(keep_models)
            & foldwise["feature_set"].isin(keep_features)
            & foldwise["calib_method"].isin(["raw", "oof_logitz", "oof_ecdf"])
        ].copy()
    if not score_ranges.empty:
        score_ranges = score_ranges[
            score_ranges["model_name"].isin(keep_models)
            & score_ranges["feature_set"].isin(keep_features)
            & score_ranges["calib_method"].isin(["raw", "oof_logitz", "oof_ecdf"])
        ].copy()
    return pooled, foldwise, score_ranges


def extract_coef_norm(fold: int) -> float:
    candidates = [
        BEST_READOUT_DIR / f"classifier_logreg_l2_raw_pipeline_fold_{fold}.joblib",
        BEST_READOUT_DIR / f"logreg_l2_fold_{fold}.joblib",
        BEST_READOUT_DIR / f"classifier_logreg_l2_fold_{fold}.joblib",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            obj = joblib.load(path)
            if hasattr(obj, "named_steps"):
                for step in reversed(list(obj.named_steps.values())):
                    if hasattr(step, "coef_"):
                        return float(np.linalg.norm(step.coef_))
            if hasattr(obj, "coef_"):
                return float(np.linalg.norm(obj.coef_))
        except Exception:
            continue
    return float("nan")


def build_classifier_hyperparameter_audit() -> pd.DataFrame:
    status = read_csv_if_exists(BEST_READOUT_DIR / "classifier_sweep_model_status.csv")
    rows = []
    if not status.empty:
        for _, row in status.iterrows():
            params = parse_jsonish(row.get("best_params"))
            c = params.get("model__C", params.get("C", np.nan))
            try:
                c = float(c)
            except Exception:
                c = np.nan
            fold = int(row.get("fold"))
            train_latent = read_csv_if_exists(BEST_READOUT_DIR / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
            test_latent = read_csv_if_exists(BEST_READOUT_DIR / "latent_cache" / f"fold_{fold}_test_latent_mu.csv")
            train_shape = f"{train_latent.shape[0]}x{train_latent.shape[1]}" if not train_latent.empty else ""
            test_shape = f"{test_latent.shape[0]}x{test_latent.shape[1]}" if not test_latent.empty else ""
            latent_dim_cols = [cname for cname in train_latent.columns if str(cname).startswith("mu_")] if not train_latent.empty else []
            n_latent_mu_features = len(latent_dim_cols)
            n_readout_features = n_latent_mu_features + 2 if n_latent_mu_features else np.nan
            coef_norm = extract_coef_norm(fold)
            rows.append(
                {
                    "audit_source": "original_classifier_sweep",
                    "fold": fold,
                    "model_name": row.get("model_name"),
                    "readout_feature_set": row.get("readout_feature_set"),
                    "trainDev_latent_cache_shape": train_shape,
                    "test_latent_cache_shape": test_shape,
                    "n_latent_mu_features": n_latent_mu_features,
                    "n_z_plus_age_sex_features": n_readout_features,
                    "selected_C": c,
                    "boundary_status": "at_original_lower_bound" if np.isfinite(c) and c <= 0.001 else "not_lower_bound",
                    "best_inner_auc": row.get("best_inner_auc"),
                    "inner_cv_context": row.get("inner_cv_context"),
                    "coef_l2_norm": coef_norm,
                    "near_null_classifier_flag": bool(np.isfinite(coef_norm) and coef_norm < 1e-6),
                    "interpretation": "Boundary hit alone is not sufficient to expand the grid; prior ultra-regularized/frozen-latent sweeps were negative. Fitted classifier coefficients were not saved in this readout folder, so coefficient norm is unavailable unless shown.",
                }
            )
    dense = read_csv_if_exists(LOGREG_C_GRID_AUDIT)
    if not dense.empty:
        subset = dense[dense["model_key"].astype(str).str.contains("recover|latent384|035", case=False, na=False)].copy()
        if subset.empty:
            subset = dense.copy()
        for _, row in subset.iterrows():
            rows.append(
                {
                    "audit_source": "dense_C_grid_sensitivity",
                    "fold": row.get("fold"),
                    "model_name": row.get("model"),
                    "readout_feature_set": row.get("readout_feature_set"),
                    "trainDev_latent_cache_shape": "",
                    "test_latent_cache_shape": "",
                    "n_latent_mu_features": np.nan,
                    "n_z_plus_age_sex_features": np.nan,
                    "selected_C": row.get("selected_C"),
                    "boundary_status": row.get("boundary_status"),
                    "best_inner_auc": row.get("best_inner_auc"),
                    "inner_cv_context": row.get("inner_cv_context"),
                    "coef_l2_norm": np.nan,
                    "near_null_classifier_flag": np.nan,
                    "interpretation": "Sensitivity-only dense C audit; not promoted unless pooled AUC/PR-AUC and subgroup behavior improve.",
                }
            )
    return pd.DataFrame(rows)


def build_cohort_summary(ledger: pd.DataFrame, summary: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"section": "counts", "metric": "base_v5_1b_tensor_subjects", "value": summary.get("base_v5_1b_tensor_subjects")},
        {"section": "counts", "metric": "all_eligible_tensor_subjects", "value": summary.get("all_eligible_tensor_subjects")},
        {"section": "counts", "metric": "all_eligible_metadata_subjects", "value": summary.get("all_eligible_metadata_subjects")},
        {"section": "counts", "metric": "matched_tensor_metadata_subjects", "value": summary.get("matched_tensor_metadata_subjects")},
        {"section": "counts", "metric": "included_subjects", "value": int((ledger["inclusion_status"] == "included").sum())},
        {"section": "counts", "metric": "excluded_subjects", "value": int((ledger["inclusion_status"] == "excluded").sum())},
    ]
    for dx, n in summary.get("vae_pool_counts", {}).items():
        rows.append({"section": "vae_pool", "metric": str(dx), "value": n})
    for dx, n in summary.get("classifier_pool_counts", {}).items():
        rows.append({"section": "classifier_pool", "metric": str(dx), "value": n})
    for reason, n in ledger.loc[ledger["inclusion_status"].eq("excluded"), "reason_code"].value_counts(dropna=False).items():
        rows.append({"section": "exclusions", "metric": str(reason), "value": int(n)})
    for sid in ["035_S_6927", "128_S_2002"]:
        sub = ledger.loc[ledger["SubjectID"].eq(sid)]
        if not sub.empty:
            r = sub.iloc[0]
            rows.append(
                {
                    "section": "required_subject_decision",
                    "metric": sid,
                    "value": f"{r.get('inclusion_status')} | {r.get('reason_code')} | VAE={r.get('in_vae_pool')} | classifier={r.get('in_classifier_pool')}",
                }
            )
    return pd.DataFrame(rows)


def write_recommendations(summary: dict[str, Any], pooled: pd.DataFrame, tensor_qc: pd.DataFrame) -> None:
    target = pooled[
        (pooled.get("model_name") == "logreg_l2_original")
        & (pooled.get("feature_set") == "z_plus_age_sex")
        & (pooled.get("calib_method") == "oof_ecdf")
        & (pooled.get("threshold_strategy") == "inner_oof_target_sens_ge_0p70_max_spec")
    ]
    if target.empty:
        target = pooled[
            (pooled.get("model_name") == "logreg_l2_original")
            & (pooled.get("feature_set") == "z_plus_age_sex")
            & (pooled.get("threshold_strategy") == "inner_oof_target_sens_ge_0p70_max_spec")
        ].head(1)
    metrics_line = "Best-model pooled metrics were not found in the calibration table."
    if not target.empty:
        r = target.iloc[0]
        metrics_line = (
            f"Best all-eligible CV readout target: AUC={r.get('auc'):.6f}, "
            f"PR-AUC={r.get('pr_auc'):.6f}, BA={r.get('balanced_accuracy'):.6f}, "
            f"Sens={r.get('sensitivity'):.6f}, Spec={r.get('specificity'):.6f}, "
            f"F1={r.get('f1'):.6f} "
            f"({r.get('model_name')}, {r.get('feature_set')}, {r.get('calib_method')}, "
            f"{r.get('threshold_strategy')})."
        )

    rec = f"""# Final Model Recommendation

## Decision

Prepare the final all-eligible ADNI model from the v5.1c recovered cohort, but do not train it in this audit package.

The all-eligible cohort is QC/metadata based: 035_S_6927 is included because signal and metadata are complete, and 128_S_2002 is excluded because diagnosis/demographics and signal provenance remain unresolved. No subject is excluded based on model performance.

## Cohort

- Clean all-eligible tensor subjects: {summary.get('all_eligible_tensor_subjects')}
- Clean all-eligible metadata rows: {summary.get('all_eligible_metadata_subjects')}
- Matched tensor+metadata subjects: {summary.get('matched_tensor_metadata_subjects')}
- VAE pool: {summary.get('vae_pool_counts')}
- Classifier pool: {summary.get('classifier_pool_counts')}

## Current Best Cross-Validated Model

{metrics_line}

The CV model source is `{rel(BEST_RUN_DIR)}` with Stage B/OOF score calibration summaries from `{rel(BEST_CALIB_DIR)}`. OOF score harmonization is treated as post-hoc fold-score calibration, not as VAE checkpoint selection or model training leakage.

## Final Model Preparation Plan

1. Train one final VAE on all eligible CN/MCI/AD subjects in the v5.1c recovered cohort using the same selected channels [1,0,2] and the best-model hyperparameters documented in this package.
2. Extract latent mu for all eligible CN/AD classifier subjects.
3. Train the frozen final classifier on latent mu + Age + Sex for all eligible CN/AD subjects.
4. Freeze preprocessing normalization/scalers, VAE weights, classifier coefficients, and threshold metadata.
5. Preserve two threshold options as distinct, predeclared artifacts: the ADNI CV-derived threshold policy and any external OASIS calibration threshold derived only from an explicit calibration subset.

## Guardrails

- No tensor, metadata, or ledger files were modified by this audit.
- The VAE checkpoint is still selected by diagnosis-agnostic validation loss, not downstream Stage B AUC.
- The supervised classifier and thresholds are selected inside training folds in nested CV; final external threshold calibration must use calibration-only external data.
- Dense C-grid and ultra-regularized readout audits do not justify changing the Stage B classifier range for the final model.
"""
    (OUTPUT_DIR / "final_model_recommendation.md").write_text(rec, encoding="utf-8")

    readme = f"""# ADNI Best Full All-Eligible End-to-End Audit

Generated: {datetime.now().isoformat(timespec='seconds')}

This package audits the full ADNI connectivity-to-classification pipeline using the all-QC-valid, metadata-valid v5.1c recovered cohort. It is read-only with respect to source tensors, metadata, ledgers, and model artifacts.

## Key Inputs

- All-eligible tensor: `{ALL_ELIGIBLE_TENSOR}`
- All-eligible metadata: `{ALL_ELIGIBLE_METADATA}`
- Best CV run: `{rel(BEST_RUN_DIR)}`
- Best Stage B calibration package: `{rel(BEST_CALIB_DIR)}`

## Subject Policy

- Include every QC-valid subject with usable tensor and required metadata.
- Include 035_S_6927.
- Exclude 128_S_2002 because the QC/metadata ledger documents unresolved diagnosis/demographics and questionable signal/provenance.
- Do not exclude any subject based on model performance.

## Main Contents

- `subject_inclusion_ledger.csv/.md` and `cohort_exclusion_summary.csv/.md`: inclusion/exclusion ledger, reason codes, and required-subject decisions.
- `tensor_qc_summary.csv/.md`: tensor fingerprint, selected channel checks, diagonal/symmetry/finite checks.
- `channel_distribution_plots/`: selected-channel histograms, mean CN/AD matrices, and AD-CN difference maps.
- `roi_order_aal3_yeo17_mapping.csv/.md` and `yeo17_network_pair_summary.csv/.md`: ROI/network mapping and network-pair summaries.
- `split_audit.csv/.md`: fold composition and leakage checks.
- `vae_training_qc.csv/.md`: per-fold training maturity and rate-distortion summaries.
- `latent_qc.csv/.md`: latent information and scanner/manufacturer leakage summaries.
- `classifier_foldwise_metrics.csv/.md` and `classifier_pooled_metrics.csv/.md`: Stage B metrics.
- `classifier_hyperparameter_audit.csv/.md`: selected C values and boundary interpretation.
- `fold_score_scale_audit.csv/.md`: fold score ranges for raw and OOF-harmonized scores.
- `final_model_recommendation.md`: final preparation plan and guardrails.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def write_command_log(args: argparse.Namespace, started: str, files_written: list[str]) -> None:
    args_payload = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    payload = {
        "script": rel(Path(__file__)),
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "args": args_payload,
        "read_only_sources": {
            "all_eligible_tensor": str(ALL_ELIGIBLE_TENSOR),
            "all_eligible_metadata": str(ALL_ELIGIBLE_METADATA),
            "base_v5_1b_tensor": str(BASE_V51B_TENSOR),
            "best_run_dir": rel(BEST_RUN_DIR),
            "best_readout_dir": rel(BEST_READOUT_DIR),
            "best_calib_dir": rel(BEST_CALIB_DIR),
        },
        "outputs": files_written,
        "safety": {
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-sha256", action="store_true", help="Skip full tensor SHA256 hashing for faster development.")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir
    started = datetime.now().isoformat(timespec="seconds")
    ensure_dir(OUTPUT_DIR)

    required = [
        ALL_ELIGIBLE_TENSOR,
        ALL_ELIGIBLE_METADATA,
        BEST_RUN_DIR,
        BEST_READOUT_DIR,
        BEST_CALIB_DIR,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    tensor = load_tensor(ALL_ELIGIBLE_TENSOR)
    metadata = pd.read_csv(ALL_ELIGIBLE_METADATA)
    base_tensor = load_tensor(BASE_V51B_TENSOR) if BASE_V51B_TENSOR.exists() else None

    ledger, summary = build_subject_inclusion_ledger(tensor, metadata, base_tensor)
    write_table(ledger, "subject_inclusion_ledger", OUTPUT_DIR, max_md_rows=120)
    cohort_summary = build_cohort_summary(ledger, summary)
    write_table(cohort_summary, "cohort_exclusion_summary", OUTPUT_DIR, max_md_rows=None)

    tensor_qc = build_tensor_qc(tensor, metadata)
    if args.skip_sha256:
        tensor_qc["tensor_sha256"] = "SKIPPED"
    write_table(tensor_qc, "tensor_qc_summary", OUTPUT_DIR)

    channel_group = build_channel_group_summary(tensor, metadata)
    write_table(channel_group, "channel_diagnosis_manufacturer_summary", OUTPUT_DIR)

    fig_manifest = plot_tensor_figures(tensor, metadata)
    write_table(fig_manifest, "channel_distribution_plot_manifest", OUTPUT_DIR, max_md_rows=None)

    split_audit = build_split_audit(metadata)
    write_table(split_audit, "split_audit", OUTPUT_DIR, max_md_rows=160)

    vae_qc = build_vae_training_qc()
    write_table(vae_qc, "vae_training_qc", OUTPUT_DIR, max_md_rows=None)

    latent_qc = build_latent_qc()
    write_table(latent_qc, "latent_qc", OUTPUT_DIR, max_md_rows=160)

    pooled, foldwise, score_ranges = load_classifier_metrics()
    write_table(foldwise, "classifier_foldwise_metrics", OUTPUT_DIR, max_md_rows=160)
    write_table(pooled, "classifier_pooled_metrics", OUTPUT_DIR, max_md_rows=120)
    write_table(score_ranges, "fold_score_scale_audit", OUTPUT_DIR, max_md_rows=160)

    hp_audit = build_classifier_hyperparameter_audit()
    write_table(hp_audit, "classifier_hyperparameter_audit", OUTPUT_DIR, max_md_rows=120)

    # Copy compact source reports when they are useful provenance and small.
    for src_name in ["final_report.md"]:
        src = BEST_CALIB_DIR / src_name
        if src.exists():
            shutil.copy2(src, OUTPUT_DIR / f"source_{src_name}")

    write_recommendations(summary, pooled, tensor_qc)

    files_written = sorted(rel(p) for p in OUTPUT_DIR.rglob("*") if p.is_file())
    write_command_log(args, started, files_written)
    print(f"Wrote audit package to {OUTPUT_DIR}")
    print(f"Rows: ledger={len(ledger)}, tensor_qc={len(tensor_qc)}, split_audit={len(split_audit)}")


if __name__ == "__main__":
    main()
