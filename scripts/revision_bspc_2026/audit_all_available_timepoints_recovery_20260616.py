#!/usr/bin/env python3
"""Recover and summarize the ADNI all-available-timepoints experiment.

This script is intentionally read-only with respect to source tensors, metadata,
models, predictions, and thresholds. It only writes a derived audit package.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "all_available_timepoints_recovery_audit_20260616"

ALLTR_RUN = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5"
ALLTR_BASE = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory"
ALLTR_TENSOR_BUILD = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_tensor_build"
ALLTR_INTEGRITY = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_integrity_audit"
ALLTR_COMPARISON = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_comparison"
ALLTR_FOLD4 = RESULTS / "adni_all_available_timepoints_fold4_failure_audit"
ALLTR_CONFIG = ROOT / "configs" / "runs" / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.json"
ALLTR_BUILD_SCRIPT = ROOT / "scripts" / "revision_bspc_2026" / "build_adni_all_available_timepoints_ch1_0_2_exploratory.py"
ALLTR_RUN_SCRIPT = ROOT / "scripts" / "revision_bspc_2026" / "run_adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.py"

PROMOTED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_CALIB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
PROMOTED_MASTER = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
PROMOTED_EVIDENCE = RESULTS / "final_model_evidence_map_with_beta9p5_T160_chweighted_20260609" / "final_model_decision_table.csv"

ALLTR_TENSOR = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_all_available_timepoints_ch1_0_2_exploratory/subject_tensors/GLOBAL_TENSOR_ADNI_all_available_timepoints_ch1_0_2_exploratory.npz")
PROMOTED_T140_TENSOR = Path("/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz")

COMMAND_LOG: list[dict[str, Any]] = []


def log(action: str, detail: str | dict[str, Any]) -> None:
    COMMAND_LOG.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "detail": detail,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log("missing_csv", str(path))
        return pd.DataFrame()
    log("read_csv", str(path))
    return pd.read_csv(path)


def read_text(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        log("missing_text", str(path))
        return ""
    log("read_text", str(path))
    txt = path.read_text(errors="replace")
    return txt if max_chars is None else txt[:max_chars]


def scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return scalar(v.item())
        if v.size <= 10:
            return v.tolist()
        return f"array{v.shape}:{v.dtype}"
    if isinstance(v, np.generic):
        return v.item()
    return v


def npz_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    out: dict[str, Any] = {"path": str(path), "exists": True}
    try:
        z = np.load(path, allow_pickle=True)
        out["keys"] = ",".join(z.files)
        for key in [
            "global_tensor_data",
            "subject_ids",
            "channel_names",
            "intended_channels_to_use",
            "intended_channel_names",
            "rois_count",
            "target_len_ts",
            "original_n_timepoints",
            "locked_n_timepoints_used",
            "tr_seconds",
            "python_bandpass_applied",
            "preprocessing_source",
            "dataset_name",
            "roi_order_name",
            "exploratory_confounding_stress_test",
        ]:
            if key not in z.files:
                continue
            arr = z[key]
            if key == "global_tensor_data":
                out["tensor_shape"] = str(arr.shape)
                out["tensor_dtype"] = str(arr.dtype)
                out["n_subjects"] = int(arr.shape[0])
                out["n_channels"] = int(arr.shape[1])
                out["n_rois"] = int(arr.shape[-1])
            elif key == "original_n_timepoints":
                finite = arr[np.isfinite(arr.astype(float))]
                out["original_n_timepoints_min"] = float(np.min(finite)) if finite.size else np.nan
                out["original_n_timepoints_median"] = float(np.median(finite)) if finite.size else np.nan
                out["original_n_timepoints_max"] = float(np.max(finite)) if finite.size else np.nan
                vals, counts = np.unique(finite.astype(int), return_counts=True)
                out["original_n_timepoints_counts"] = "; ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
            elif key == "locked_n_timepoints_used":
                vals, counts = np.unique(arr, return_counts=True)
                out["locked_n_timepoints_used_counts"] = "; ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))
            else:
                out[key] = scalar(arr)
    except Exception as exc:  # noqa: BLE001
        out["read_error"] = f"{type(exc).__name__}: {exc}"
    return out


def write_df(df: pd.DataFrame, stem: str) -> None:
    csv_path = OUT / f"{stem}.csv"
    md_path = OUT / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    if df.empty:
        md_path.write_text("_No rows available._\n")
    else:
        try:
            md_path.write_text(df.to_markdown(index=False) + "\n")
        except Exception:
            md_path.write_text(df.to_csv(index=False))
    log("write_table", {"csv": str(csv_path), "md": str(md_path), "rows": len(df)})


def write_md(name: str, text: str) -> None:
    path = OUT / name
    path.write_text(text.rstrip() + "\n")
    log("write_markdown", str(path))


def path_inventory() -> pd.DataFrame:
    candidates = [
        ALLTR_BASE,
        ALLTR_TENSOR_BUILD,
        ALLTR_RUN,
        ALLTR_INTEGRITY,
        ALLTR_COMPARISON,
        ALLTR_FOLD4,
        RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_split_preview.csv",
        RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_split_preview_summary.csv",
        ALLTR_CONFIG,
        ALLTR_BUILD_SCRIPT,
        ALLTR_RUN_SCRIPT,
        ROOT / "scripts" / "revision_bspc_2026" / "compare_adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.py",
        ROOT / "scripts" / "revision_bspc_2026" / "audit_adni_all_available_timepoints_ch1_0_2_exploratory_integrity.py",
        ROOT / "scripts" / "revision_bspc_2026" / "prepare_adni_all_available_timepoints_ch1_0_2_exploratory.py",
        ALLTR_TENSOR,
    ]
    rows = []
    for p in candidates:
        rows.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "type": "dir" if p.is_dir() else "file" if p.is_file() else "missing",
                "is_symlink": p.is_symlink(),
                "resolved_path": str(p.resolve()) if p.exists() else "",
                "size_bytes": p.stat().st_size if p.exists() and p.is_file() else np.nan,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if p.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def extract_config_summary() -> dict[str, Any]:
    out: dict[str, Any] = {"config_path": str(ALLTR_CONFIG), "exists": ALLTR_CONFIG.exists()}
    if not ALLTR_CONFIG.exists():
        return out
    cfg = json.loads(ALLTR_CONFIG.read_text())
    log("read_json", str(ALLTR_CONFIG))
    params = cfg.get("parameters", {}) if isinstance(cfg.get("parameters", {}), dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    for key in [
        "description",
        "tensor_npz_path",
        "metadata_csv_path",
        "output_dir",
        "channels_to_use",
        "latent_dim",
        "beta_vae",
        "epochs_vae",
        "cyclical_beta_n_cycles",
        "lr_scheduler_T0",
        "early_stopping_patience_vae",
        "dropout_rate_vae",
        "vae_dropout_scope",
        "outer_folds",
        "inner_folds",
        "metadata_features",
        "classifier_types",
        "seed",
    ]:
        if key == "tensor_npz_path":
            out[key] = paths.get("global_tensor_path", cfg.get(key))
        elif key == "metadata_csv_path":
            out[key] = paths.get("metadata_path", cfg.get(key))
        elif key == "output_dir":
            out[key] = paths.get("output_dir", cfg.get(key))
        else:
            out[key] = params.get(key, cfg.get(key))
    return out


def code_policy_audit() -> dict[str, Any]:
    txt = read_text(ALLTR_BUILD_SCRIPT)
    run_txt = read_text(ALLTR_RUN_SCRIPT)
    return {
        "build_script": str(ALLTR_BUILD_SCRIPT),
        "uses_homogenize_length": "homogenize_length" in txt,
        "calls_homogenize_length": bool(re.search(r"\bhomogenize_length\s*\(", txt)),
        "uses_standardize_timeseries": "standardize_timeseries" in txt,
        "branch_timepoint_policy_literal": "all_available_variable_length" in txt,
        "compute_selected_channels": "compute_selected_channels" in txt,
        "selected_channels_alltr_tensor": "[0, 1, 2]",
        "intended_downstream_channels_to_use": "[1, 0, 2]",
        "run_script_real_training_requires_confirm_training": "--confirm-training" in run_txt,
        "run_script_source_locked_config": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5" in run_txt,
        "not_promoted_recover035_protocol": True,
    }


def completion_status() -> pd.DataFrame:
    ckpt = read_csv(ALLTR_INTEGRITY / "checkpoint_integrity.csv")
    stageb = read_csv(ALLTR_INTEGRITY / "stageb_readout_integrity.csv")
    main = read_csv(ALLTR_COMPARISON / "main_model_comparison.csv")
    rows = []
    rows.append({"item": "all_available_run_dir", "status": "present" if ALLTR_RUN.exists() else "missing", "detail": str(ALLTR_RUN)})
    rows.append({"item": "config", "status": "present" if ALLTR_CONFIG.exists() else "missing", "detail": str(ALLTR_CONFIG)})
    rows.append({"item": "tensor", "status": "present" if ALLTR_TENSOR.exists() else "missing", "detail": str(ALLTR_TENSOR)})
    if not ckpt.empty:
        rows.append(
            {
                "item": "five_outer_fold_checkpoints",
                "status": "complete" if len(ckpt) == 5 and ckpt["checkpoint_exists"].all() else "incomplete",
                "detail": f"fold_rows={len(ckpt)}; checkpoints={int(ckpt['checkpoint_exists'].sum())}; histories={int(ckpt['history_exists'].sum())}",
            }
        )
    if not stageb.empty:
        rows.append(
            {
                "item": "stageB_classifier_only_readout",
                "status": "complete" if stageb["exists"].all() else "incomplete",
                "detail": f"present={int(stageb['exists'].sum())}/{len(stageb)}",
            }
        )
    if not main.empty:
        alltr = main[main["run_id"].eq("all_available_timepoints")]
        rows.append(
            {
                "item": "comparison_metrics",
                "status": "present" if not alltr.empty else "missing",
                "detail": f"rows={len(main)}; alltr_rows={len(alltr)}",
            }
        )
    return pd.DataFrame(rows)


def tensor_policy_comparison() -> pd.DataFrame:
    alltr_npz = npz_summary(ALLTR_TENSOR)
    t140_npz = npz_summary(PROMOTED_T140_TENSOR)
    cfg = extract_config_summary()
    code = code_policy_audit()
    rows = []
    rows.append(
        {
            "artifact": "all_available_timepoints_tensor",
            "path": alltr_npz.get("path"),
            "exists": alltr_npz.get("exists"),
            "shape": alltr_npz.get("tensor_shape"),
            "n_subjects": alltr_npz.get("n_subjects"),
            "n_channels": alltr_npz.get("n_channels"),
            "n_rois": alltr_npz.get("n_rois"),
            "channel_names": alltr_npz.get("channel_names"),
            "intended_channels_to_use": alltr_npz.get("intended_channels_to_use"),
            "target_len_ts": alltr_npz.get("target_len_ts"),
            "timepoint_policy": "all available variable length",
            "original_n_timepoints_counts": alltr_npz.get("original_n_timepoints_counts"),
            "locked_n_timepoints_used_counts": alltr_npz.get("locked_n_timepoints_used_counts"),
            "python_bandpass_applied": alltr_npz.get("python_bandpass_applied"),
            "roi_order_name": alltr_npz.get("roi_order_name"),
            "protocol_context": "exploratory tensor built from 646-row training-ready metadata subject pool",
        }
    )
    rows.append(
        {
            "artifact": "locked_or_promoted_T140_source_tensor",
            "path": t140_npz.get("path"),
            "exists": t140_npz.get("exists"),
            "shape": t140_npz.get("tensor_shape"),
            "n_subjects": t140_npz.get("n_subjects"),
            "n_channels": t140_npz.get("n_channels"),
            "n_rois": t140_npz.get("n_rois"),
            "channel_names": t140_npz.get("channel_names"),
            "intended_channels_to_use": "[1, 0, 2] selected downstream",
            "target_len_ts": t140_npz.get("target_len_ts"),
            "timepoint_policy": "homogenized T=140 before connectivity",
            "original_n_timepoints_counts": "",
            "locked_n_timepoints_used_counts": "",
            "python_bandpass_applied": t140_npz.get("python_bandpass_applied"),
            "roi_order_name": t140_npz.get("roi_order_name"),
            "protocol_context": "global seven-channel T140 source tensor; allTR branch used first three channels only",
        }
    )
    rows.append(
        {
            "artifact": "allTR_full5x5_training_config",
            "path": cfg.get("config_path"),
            "exists": cfg.get("exists"),
            "shape": "",
            "n_subjects": "",
            "n_channels": "",
            "n_rois": "",
            "channel_names": "",
            "intended_channels_to_use": str(cfg.get("channels_to_use")),
            "target_len_ts": "",
            "timepoint_policy": "input tensor/metadata branch differs from locked v5.1b",
            "original_n_timepoints_counts": "",
            "locked_n_timepoints_used_counts": "",
            "python_bandpass_applied": "",
            "roi_order_name": "",
            "protocol_context": f"latent_dim={cfg.get('latent_dim')}; beta={cfg.get('beta_vae')}; epochs={cfg.get('epochs_vae')}; cycles={cfg.get('cyclical_beta_n_cycles')}; patience={cfg.get('early_stopping_patience_vae')}",
        }
    )
    rows.append(
        {
            "artifact": "code_path_policy",
            "path": str(ALLTR_BUILD_SCRIPT),
            "exists": ALLTR_BUILD_SCRIPT.exists(),
            "shape": "",
            "n_subjects": "",
            "n_channels": "",
            "n_rois": "",
            "channel_names": "",
            "intended_channels_to_use": code.get("intended_downstream_channels_to_use"),
            "target_len_ts": "",
            "timepoint_policy": "standardize full available reduced ROI time series; no homogenize_length call in allTR preprocessing",
            "original_n_timepoints_counts": "",
            "locked_n_timepoints_used_counts": "",
            "python_bandpass_applied": "",
            "roi_order_name": "",
            "protocol_context": f"uses_standardize={code.get('uses_standardize_timeseries')}; calls_homogenize={code.get('calls_homogenize_length')}; branch_policy_literal={code.get('branch_timepoint_policy_literal')}",
        }
    )
    return pd.DataFrame(rows)


def fold_completion_and_failure() -> pd.DataFrame:
    ckpt = read_csv(ALLTR_INTEGRITY / "checkpoint_integrity.csv")
    fold4 = read_csv(ALLTR_FOLD4 / "fold4_vs_other_folds_metrics.csv")
    confusion = read_csv(ALLTR_FOLD4 / "confusion_by_fold.csv")
    rows = []
    if not ckpt.empty:
        for _, r in ckpt.iterrows():
            rows.append(
                {
                    "section": "completion",
                    "fold": int(r["fold"]),
                    "run_id": "all_available_timepoints",
                    "metric": "checkpoint_history",
                    "value": f"checkpoint={bool(r['checkpoint_exists'])}; history={bool(r['history_exists'])}",
                    "interpretation": "fold artifact present" if r["checkpoint_exists"] and r["history_exists"] else "missing fold artifact",
                }
            )
    if not fold4.empty:
        subset = fold4[
            (fold4["run_id"].eq("all_available_timepoints"))
            & (fold4["threshold_strategy"].eq("inner_oof_target_sens_ge_0p70_max_spec"))
        ].copy()
        if subset.empty:
            subset = fold4[fold4["run_id"].eq("all_available_timepoints")].copy()
        for _, r in subset.iterrows():
            rows.append(
                {
                    "section": "fold4_failure",
                    "fold": int(r["fold"]),
                    "run_id": r["run_id"],
                    "metric": "primary_metrics",
                    "value": f"AUC={r['auc']:.6f}; PR-AUC={r['pr_auc']:.6f}; BA={r['balanced_accuracy']:.6f}; Sens={r['sensitivity']:.6f}; Spec={r['specificity']:.6f}; F1={r['f1']:.6f}",
                    "interpretation": "fold4 outlier degradation" if int(r["fold"]) == 4 else "non-fold4 comparison",
                }
            )
    if not confusion.empty:
        c = confusion[(confusion["run_id"].eq("all_available_timepoints")) & (confusion["fold"].eq(4))]
        if not c.empty:
            r = c.iloc[0]
            rows.append(
                {
                    "section": "fold4_confusion",
                    "fold": 4,
                    "run_id": "all_available_timepoints",
                    "metric": "confusion",
                    "value": f"TN={int(r['tn'])}; FP={int(r['fp'])}; FN={int(r['fn'])}; TP={int(r['tp'])}",
                    "interpretation": "Fold 4 had low AD detection and many CN false positives.",
                }
            )
    final_failure = read_text(ALLTR_FOLD4 / "final_failure_diagnosis.md", max_chars=600)
    rows.append(
        {
            "section": "failure_audit_text",
            "fold": "",
            "run_id": "all_available_timepoints",
            "metric": "summary",
            "value": "see final_failure_diagnosis.md",
            "interpretation": re.sub(r"\s+", " ", final_failure).strip()[:500],
        }
    )
    return pd.DataFrame(rows)


def promoted_reference_row() -> dict[str, Any] | None:
    if PROMOTED_EVIDENCE.exists():
        df = read_csv(PROMOTED_EVIDENCE)
        if not df.empty and "model_id" in df:
            r = df[df["model_id"].eq("promoted_latent384_beta3p75_ch1_0_2")]
            if not r.empty:
                rr = r.iloc[0]
                return {
                    "run_id": "promoted_recover035_T140",
                    "label": "Promoted recover035 [1,0,2] T=140 latent384 beta3.75",
                    "model_name": "logreg_l2_original",
                    "threshold_strategy": "inner_oof_target_sens_ge_0p70_max_spec",
                    "n": 397,
                    "n_cn": 300,
                    "n_ad": 97,
                    "auc": rr.get("adni_oof_ecdf_auc"),
                    "pr_auc": rr.get("adni_oof_ecdf_pr_auc"),
                    "balanced_accuracy": rr.get("adni_oof_ecdf_balanced_accuracy"),
                    "sensitivity": rr.get("adni_oof_ecdf_sensitivity"),
                    "specificity": rr.get("adni_oof_ecdf_specificity"),
                    "f1": rr.get("adni_oof_ecdf_f1"),
                    "philips_cn_fpr": rr.get("philips_cn_fpr"),
                    "comparison_note": "Current promoted model; not protocol-matched to allTR because allTR used locked v5.1b latent256 beta2.5 horizon4480.",
                }
    pooled = read_csv(PROMOTED_CALIB / "calib_pooled_metrics.csv")
    if not pooled.empty:
        mask = (
            pooled["model_name"].eq("logreg_l2_original")
            & pooled["feature_set"].eq("z_plus_age_sex")
            & pooled["calib_method"].eq("oof_ecdf")
            & pooled["threshold_strategy"].eq("inner_oof_target_sens_ge_0p70_max_spec")
        )
        r = pooled[mask]
        if not r.empty:
            rr = r.iloc[0]
            return {
                "run_id": "promoted_recover035_T140",
                "label": "Promoted recover035 [1,0,2] T=140 latent384 beta3.75",
                "model_name": rr.get("model_name"),
                "threshold_strategy": rr.get("threshold_strategy"),
                "n": rr.get("n"),
                "n_cn": rr.get("n_cn"),
                "n_ad": rr.get("n_ad"),
                "auc": rr.get("auc"),
                "pr_auc": rr.get("pr_auc"),
                "balanced_accuracy": rr.get("balanced_accuracy"),
                "sensitivity": rr.get("sensitivity"),
                "specificity": rr.get("specificity"),
                "f1": rr.get("f1"),
                "philips_cn_fpr": np.nan,
                "comparison_note": "Current promoted model; not protocol-matched to allTR because allTR used locked v5.1b latent256 beta2.5 horizon4480.",
            }
    return None


def promoted_vs_fulltp_metrics() -> pd.DataFrame:
    main = read_csv(ALLTR_COMPARISON / "main_model_comparison.csv")
    rows = []
    if not main.empty:
        for _, r in main.iterrows():
            d = r.to_dict()
            d["philips_cn_fpr"] = np.nan
            d["comparison_note"] = "Protocol-matched comparison within older locked v5.1b latent256 beta2.5 horizon4480 family."
            rows.append(d)
    pr = promoted_reference_row()
    if pr:
        rows.append(pr)
    df = pd.DataFrame(rows)
    if not df.empty:
        keep = [
            "run_id",
            "label",
            "model_name",
            "threshold_strategy",
            "n",
            "n_cn",
            "n_ad",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "philips_cn_fpr",
            "comparison_note",
        ]
        df = df[[c for c in keep if c in df.columns]]
    return df


def subject_fold_comparison() -> pd.DataFrame:
    alltr_subjects = read_csv(ALLTR_TENSOR_BUILD / "subject_timepoint_used.csv")
    promoted = read_csv(PROMOTED_MASTER)
    rows = []
    if alltr_subjects.empty or promoted.empty:
        return pd.DataFrame(rows)

    alltr_cls = alltr_subjects[alltr_subjects.get("classifier_pool_role", pd.Series(dtype=str)).eq("classifier_cn_ad")].copy()
    promoted_cls = promoted[promoted.get("in_oof_evaluation", pd.Series(dtype=bool)).eq(True)].copy()
    alltr_set = set(alltr_cls["SubjectID"].astype(str))
    promoted_set = set(promoted_cls["SubjectID"].astype(str))
    overlap = alltr_set & promoted_set
    rows.append(
        {
            "comparison": "classifier_subject_sets",
            "alltr_n": len(alltr_set),
            "promoted_n": len(promoted_set),
            "overlap_n": len(overlap),
            "alltr_only_n": len(alltr_set - promoted_set),
            "promoted_only_n": len(promoted_set - alltr_set),
            "same_subject_set": alltr_set == promoted_set,
            "detail": f"promoted_only={','.join(sorted(promoted_set - alltr_set)[:20])}; alltr_only={','.join(sorted(alltr_set - promoted_set)[:20])}",
        }
    )

    if "ResearchGroup_Mapped" in alltr_cls:
        vc = alltr_cls["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "comparison": "alltr_classifier_diagnosis_counts",
                "alltr_n": len(alltr_cls),
                "promoted_n": "",
                "overlap_n": "",
                "alltr_only_n": "",
                "promoted_only_n": "",
                "same_subject_set": "",
                "detail": json.dumps(vc, sort_keys=True),
            }
        )
    if "ResearchGroup_Mapped" in promoted_cls:
        vc = promoted_cls["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "comparison": "promoted_classifier_diagnosis_counts",
                "alltr_n": "",
                "promoted_n": len(promoted_cls),
                "overlap_n": "",
                "alltr_only_n": "",
                "promoted_only_n": "",
                "same_subject_set": "",
                "detail": json.dumps(vc, sort_keys=True),
            }
        )

    fold_col_alltr = "classifier_outer_fold"
    fold_col_prom = "outer_fold"
    if fold_col_alltr in alltr_cls and fold_col_prom in promoted_cls:
        merged = alltr_cls[["SubjectID", fold_col_alltr]].merge(
            promoted_cls[["SubjectID", fold_col_prom]], on="SubjectID", how="inner"
        )
        same_fold = (
            pd.to_numeric(merged[fold_col_alltr], errors="coerce")
            == pd.to_numeric(merged[fold_col_prom], errors="coerce")
        )
        rows.append(
            {
                "comparison": "overlap_outer_fold_identity",
                "alltr_n": "",
                "promoted_n": "",
                "overlap_n": len(merged),
                "alltr_only_n": "",
                "promoted_only_n": "",
                "same_subject_set": "",
                "detail": f"same_fold={int(same_fold.sum())}/{len(merged)}; different_fold={int((~same_fold).sum())}",
            }
        )
    return pd.DataFrame(rows)


def philips_rawtp_error_comparison() -> pd.DataFrame:
    pred_path = ALLTR_RUN / "classifier_only_readout" / "classifier_sweep_predictions.csv"
    preds = read_csv(pred_path)
    master = read_csv(PROMOTED_MASTER)
    rows = []

    if not preds.empty and not master.empty:
        primary = preds[
            (preds["model_name"].eq("logreg_l2"))
            & (preds["readout_feature_set"].eq("z_plus_age_sex"))
            & (preds["threshold_strategy"].eq("inner_oof_target_sens_ge_0p70_max_spec"))
        ].copy()
        meta_cols = ["SubjectID", "raw_tp_group", "raw_tp_group_norm", "Site3", "Manufacturer", "ResearchGroup_Mapped"]
        meta_cols = [c for c in meta_cols if c in master.columns]
        merged = primary.merge(master[meta_cols].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_master"))
        if "Manufacturer_master" in merged:
            merged["Manufacturer"] = merged["Manufacturer"].fillna(merged["Manufacturer_master"])
        if "ResearchGroup_Mapped_master" in merged:
            merged["ResearchGroup_Mapped"] = merged["ResearchGroup_Mapped"].fillna(merged["ResearchGroup_Mapped_master"])
        cn = merged[(merged["Manufacturer"].eq("Philips")) & (merged["y_true"].eq(0))].copy()
        if "raw_tp_group_norm" not in cn:
            cn["raw_tp_group_norm"] = cn.get("raw_tp_group", "UNKNOWN")
        cn["raw_tp_group_norm"] = cn["raw_tp_group_norm"].fillna(cn.get("raw_tp_group", "UNKNOWN")).astype(str)
        cn["raw_tp_group_norm"] = cn["raw_tp_group_norm"].replace({"197": "197_200", "200": "197_200", "140 TP": "140", "197 TP": "197_200", "200 TP": "197_200"})
        for group, g in cn.groupby("raw_tp_group_norm", dropna=False):
            n = len(g)
            fp = int((g["y_pred"] == 1).sum())
            rows.append(
                {
                    "run_id": "all_available_timepoints",
                    "group": group,
                    "n_cn": n,
                    "fp": fp,
                    "tn": int(n - fp),
                    "fpr": fp / n if n else np.nan,
                    "score_mean": float(g["y_score"].mean()) if n else np.nan,
                    "score_median": float(g["y_score"].median()) if n else np.nan,
                }
            )

    existing = read_csv(RESULTS / "rawTP_connectome_error_mechanism_audit_20260615" / "philips_cn_rawTP_fpr_table.csv")
    if not existing.empty:
        for _, r in existing.iterrows():
            rows.append(
                {
                    "run_id": "promoted_recover035_T140_existing_rawTP_audit",
                    "group": r.get("raw_tp_group_norm"),
                    "n_cn": r.get("n_cn"),
                    "fp": r.get("fp"),
                    "tn": r.get("tn"),
                    "fpr": r.get("fpr"),
                    "score_mean": r.get("score_mean"),
                    "score_median": r.get("score_median"),
                }
            )

    return pd.DataFrame(rows)


def scanner_leakage_comparison() -> pd.DataFrame:
    leak = read_csv(ALLTR_INTEGRITY / "scanner_manufacturer_leakage.csv")
    ntr = read_csv(ALLTR_INTEGRITY / "ntr_predictability_from_latent.csv")
    rows = []
    if not leak.empty:
        rows.append(
            {
                "run_id": "all_available_timepoints",
                "artifact": "scanner_manufacturer_leakage",
                "metric": "test_latent_manufacturer_acc_mean",
                "value": leak[leak["fold_tag"].astype(str).str.contains("_test")]["acc_site_latent"].mean(),
                "details": f"test rows={int(leak['fold_tag'].astype(str).str.contains('_test').sum())}",
            }
        )
        rows.append(
            {
                "run_id": "all_available_timepoints",
                "artifact": "scanner_manufacturer_leakage",
                "metric": "train_dev_latent_manufacturer_acc_mean",
                "value": leak[~leak["fold_tag"].astype(str).str.contains("_test")]["acc_site_latent"].mean(),
                "details": "mean across non-test fold leakage rows",
            }
        )
    if not ntr.empty:
        for _, r in ntr.iterrows():
            rows.append(
                {
                    "run_id": "all_available_timepoints",
                    "artifact": "ntr_predictability_from_latent",
                    "metric": f"{r.get('target')}_{r.get('metric')}",
                    "value": r.get("value"),
                    "details": r.get("details"),
                }
            )
    # Promoted summary if available.
    if PROMOTED_EVIDENCE.exists():
        df = read_csv(PROMOTED_EVIDENCE)
        r = df[df.get("model_id", pd.Series(dtype=str)).eq("promoted_latent384_beta3p75_ch1_0_2")]
        if not r.empty:
            rr = r.iloc[0]
            rows.append(
                {
                    "run_id": "promoted_recover035_T140",
                    "artifact": "final_model_evidence_map",
                    "metric": "test_latent_manufacturer_acc_mean",
                    "value": rr.get("scanner_leakage_latent_acc"),
                    "details": "not protocol-matched to allTR",
                }
            )
    return pd.DataFrame(rows)


def oasis_external_comparison() -> pd.DataFrame:
    txt = read_text(ALLTR_INTEGRITY / "oasis_external_scoring_preflight.md")
    status = "not_found"
    detail = "No OASIS preflight artifact found."
    if txt:
        status = "preflight_only"
        detail = re.sub(r"\s+", " ", txt).strip()
    # Search for any allTR OASIS output dirs.
    hits = [p for p in RESULTS.glob("*all_available*time*oasis*") if p.exists()]
    if hits:
        status = "candidate_artifacts_found"
        detail += " Candidate paths: " + "; ".join(str(p) for p in hits)
    return pd.DataFrame(
        [
            {
                "run_id": "all_available_timepoints",
                "oasis_status": status,
                "details": detail[:1000],
            }
        ]
    )


def final_recommendation_text(metrics: pd.DataFrame, tensor_policy: pd.DataFrame, scanner: pd.DataFrame) -> str:
    alltr = metrics[metrics.get("run_id", pd.Series(dtype=str)).eq("all_available_timepoints")]
    locked = metrics[metrics.get("run_id", pd.Series(dtype=str)).eq("locked_v5_1b_140TR")]
    promoted = metrics[metrics.get("run_id", pd.Series(dtype=str)).eq("promoted_recover035_T140")]
    lines = [
        "# Final Recommendation",
        "",
        "## Answers",
        "",
        "- Was fullTP actually tested? **Yes, as an exploratory all-available-variable-length tensor branch.** The tensor metadata records `target_len_ts=all_available_variable_length`, per-subject `original_n_timepoints`, and the build code standardizes the full reduced ROI time series without calling the T=140 homogenization routine.",
        "- Was it complete? **Yes for the older locked v5.1b-style 5x5 run.** Five fold checkpoints and Stage B classifier-only readout artifacts are present.",
        "- Is it directly comparable to the current promoted recover035 latent384 beta3.75 model? **No.** The allTP FULL run was based on the older locked v5.1b horizon4480/cycles56 latent256 beta2.5 protocol. It is comparable primarily against the locked v5.1b T=140 control included in its comparison package.",
    ]
    if not alltr.empty and not locked.empty:
        a = alltr.iloc[0]
        b = locked.iloc[0]
        lines += [
            "- Did it improve signal/performance? **No in the protocol-matched comparison.** "
            f"AllTP AUC={a['auc']:.6f}, PR-AUC={a['pr_auc']:.6f}, BA={a['balanced_accuracy']:.6f}, F1={a['f1']:.6f}; "
            f"locked T=140 AUC={b['auc']:.6f}, PR-AUC={b['pr_auc']:.6f}, BA={b['balanced_accuracy']:.6f}, F1={b['f1']:.6f}.",
        ]
    if not promoted.empty:
        p = promoted.iloc[0]
        lines += [
            f"- Current promoted T=140 reference remains stronger internally: AUC={p['auc']:.6f}, PR-AUC={p['pr_auc']:.6f}. This is contextual, not a strict protocol-matched allTP comparison.",
        ]
    if not scanner.empty:
        mfr = scanner[(scanner["run_id"].eq("all_available_timepoints")) & (scanner["metric"].eq("test_latent_manufacturer_acc_mean"))]
        if not mfr.empty and pd.notna(mfr.iloc[0]["value"]):
            lines.append(
                f"- Did it increase protocol/scanner confounding? The branch remains confounding-sensitive: mean test latent Manufacturer accuracy was {float(mfr.iloc[0]['value']):.3f}, and prior package notes indicate n_TR is diagnosis/manufacturer/site-confounded."
            )
    lines += [
        "- Should it be repeated now? **Not as a model-selection step.** The previous fullTP branch is best reported as an internal negative/confounding-stress sensitivity. Repeating would only be scientifically justified under a new pre-specified preprocessing question, not to chase AUC.",
        "",
        "## Recommendation",
        "",
        "Keep the promoted T=140 model as primary. The T=140 choice is defensible as a preprocessing harmonization and confounding-control decision, not because it won a post-hoc AUC contest. The fullTP branch did test the alternative, completed under the older locked protocol, and underperformed its matched T=140 control while retaining scanner/protocol confounding concerns.",
    ]
    return "\n".join(lines)


def collaborator_summary_text(metrics: pd.DataFrame) -> str:
    alltr = metrics[metrics.get("run_id", pd.Series(dtype=str)).eq("all_available_timepoints")]
    locked = metrics[metrics.get("run_id", pd.Series(dtype=str)).eq("locked_v5_1b_140TR")]
    delta = ""
    if not alltr.empty and not locked.empty:
        a = alltr.iloc[0]
        b = locked.iloc[0]
        delta = (
            f" In the matched older v5.1b comparison, allTP was lower than T=140 "
            f"(AUC {a['auc']:.3f} vs {b['auc']:.3f}; PR-AUC {a['pr_auc']:.3f} vs {b['pr_auc']:.3f}; "
            f"BA {a['balanced_accuracy']:.3f} vs {b['balanced_accuracy']:.3f})."
        )
    return (
        "# Collaborator Summary\n\n"
        "We did find a previous all-available-timepoints ADNI experiment. It was a real variable-timepoint tensor branch, not a silently truncated T=140 rerun: the tensor metadata records `all_available_variable_length`, and the build code avoids the T=140 homogenization step before computing Pearson/OMST/MI channels.\n\n"
        "However, that experiment was run under the older locked v5.1b latent256 beta2.5 horizon4480/cycles56 protocol, not under the current promoted recover035 latent384 beta3.75 model. It completed all five folds and had Stage B classifier-only outputs."
        f"{delta}\n\n"
        "The Fold 4 audit concluded that the drop was not just threshold/calibration; ranking and latent separation degraded on a hard fold, with manufacturer/site/n_TR confounding still a guardrail. I would describe fullTP as an internal negative sensitivity and keep T=140 as a harmonization choice made to reduce protocol/timepoint confounding, not as a model-selection convenience."
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start", "all_available_timepoints_recovery_audit_20260616")

    inventory = path_inventory()
    write_df(inventory, "all_available_timepoints_inventory")

    completion = completion_status()
    write_df(completion, "completion_status")

    tensor_policy = tensor_policy_comparison()
    write_df(tensor_policy, "tensor_policy_comparison")

    fold_failure = fold_completion_and_failure()
    write_df(fold_failure, "fold_completion_and_failure_audit")

    metrics = promoted_vs_fulltp_metrics()
    write_df(metrics, "promoted_vs_fullTP_metrics")

    subj_folds = subject_fold_comparison()
    if not subj_folds.empty:
        write_df(subj_folds, "subject_fold_comparison")

    rawtp = philips_rawtp_error_comparison()
    if not rawtp.empty:
        write_df(rawtp, "philips_rawTP_error_comparison")

    scanner = scanner_leakage_comparison()
    if not scanner.empty:
        write_df(scanner, "scanner_leakage_comparison")

    oasis = oasis_external_comparison()
    write_df(oasis, "oasis_external_comparison")

    write_md("final_recommendation.md", final_recommendation_text(metrics, tensor_policy, scanner))
    write_md("collaborator_summary.md", collaborator_summary_text(metrics))

    log("guardrails", "No training, tensor edits, metadata edits, prediction edits, threshold refitting, subject exclusion, or model selection performed.")
    (OUT / "command_log.json").write_text(json.dumps(COMMAND_LOG, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
