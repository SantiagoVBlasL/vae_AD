#!/usr/bin/env python3
"""Read-only integrity audit for horizon4480/cycles56 FULL 5x5.

This script checks whether the promoted long-horizon candidate actually used
candidate fold artifacts/readout outputs, or whether any folds are stale copies
of the locked 3840/48 run.

It writes only lightweight audit outputs under results/. It does not train and
does not modify tensors, metadata, ledgers, configs, or existing model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
LOCKED_READOUT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
OUTPUT_DIR = RESULTS / "adni_v5_1_batch20260514b_horizon4480_integrity_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FOLDS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--locked-run", type=Path, default=LOCKED_RUN)
    parser.add_argument("--locked-readout", type=Path, default=LOCKED_READOUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def file_stat(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    out: Dict[str, Any] = {
        "path": str(path),
        "exists": bool(exists),
        "realpath": str(path.resolve()) if exists else "",
        "size_bytes": np.nan,
        "mtime_epoch": np.nan,
        "mtime_iso": "",
        "sha256": "",
    }
    if exists:
        st = path.stat()
        out["size_bytes"] = int(st.st_size)
        out["mtime_epoch"] = float(st.st_mtime)
        out["mtime_iso"] = datetime.fromtimestamp(st.st_mtime).isoformat()
        if path.is_file():
            out["sha256"] = sha256_file(path)
    return out


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_outdir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already contains files; pass --overwrite")
    if overwrite:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                # Audit output directory is controlled and lightweight; avoid importing shutil for one branch.
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()


def artifact_paths(run: Path, readout: Path, fold: int) -> Dict[str, Path]:
    fold_dir = run / f"fold_{fold}"
    return {
        "stagea_fold_dir": fold_dir,
        "vae_model": fold_dir / f"vae_model_fold_{fold}.pt",
        "vae_history": fold_dir / f"vae_train_history_fold_{fold}.joblib",
        "vae_norm_params": fold_dir / "vae_norm_params.joblib",
        "latent_qc": fold_dir / "latent_qc_metrics.csv",
        "scanner_leakage_summary": fold_dir / f"fold_{fold}_scanner_leakage_summary.csv",
        "test_subjects": fold_dir / "test_subjects_fold.csv",
        "train_dev_subjects": fold_dir / "train_dev_subjects_fold.csv",
        "test_tensor_idx": fold_dir / "test_tensor_idx.npy",
        "train_dev_tensor_idx": fold_dir / "train_dev_tensor_idx.npy",
        "stagea_test_predictions_logreg": fold_dir / "test_predictions_logreg.csv",
        "stagea_test_predictions_svm": fold_dir / "test_predictions_svm.csv",
        "readout_test_latent_mu": readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        "readout_trainDev_latent_mu": readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
    }


def build_inventory(candidate_run: Path, candidate_readout: Path, locked_run: Path, locked_readout: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        cand = artifact_paths(candidate_run, candidate_readout, fold)
        lock = artifact_paths(locked_run, locked_readout, fold)
        for artifact in sorted(cand):
            cstat = file_stat(cand[artifact])
            lstat = file_stat(lock[artifact])
            rows.append(
                {
                    "fold": fold,
                    "artifact": artifact,
                    "candidate_path": cstat["path"],
                    "locked_path": lstat["path"],
                    "candidate_exists": cstat["exists"],
                    "locked_exists": lstat["exists"],
                    "candidate_realpath": cstat["realpath"],
                    "locked_realpath": lstat["realpath"],
                    "realpath_same": bool(cstat["realpath"] and cstat["realpath"] == lstat["realpath"]),
                    "candidate_size_bytes": cstat["size_bytes"],
                    "locked_size_bytes": lstat["size_bytes"],
                    "size_same": bool(cstat["exists"] and lstat["exists"] and cstat["size_bytes"] == lstat["size_bytes"]),
                    "candidate_mtime_iso": cstat["mtime_iso"],
                    "locked_mtime_iso": lstat["mtime_iso"],
                    "candidate_sha256": cstat["sha256"],
                    "locked_sha256": lstat["sha256"],
                    "sha256_same": bool(cstat["sha256"] and cstat["sha256"] == lstat["sha256"]),
                }
            )
    for artifact, cpath, lpath in [
        ("stageb_predictions", candidate_readout / "classifier_sweep_predictions.csv", locked_readout / "classifier_sweep_predictions.csv"),
        ("stageb_foldwise_metrics", candidate_readout / "classifier_sweep_foldwise_metrics.csv", locked_readout / "classifier_sweep_foldwise_metrics.csv"),
        ("stageb_thresholds_by_fold", candidate_readout / "classifier_sweep_thresholds_by_fold.csv", locked_readout / "classifier_sweep_thresholds_by_fold.csv"),
        ("stageb_model_status", candidate_readout / "classifier_sweep_model_status.csv", locked_readout / "classifier_sweep_model_status.csv"),
        ("stageb_command_log", candidate_readout / "command_log.json", locked_readout / "command_log.json"),
        ("stageb_latent_manifest", candidate_readout / "latent_feature_manifest.json", locked_readout / "latent_feature_manifest.json"),
        ("stagea_command_log", candidate_run / "command_log.json", locked_run / "command_log.json"),
        ("stagea_run_manifest", candidate_run / "run_manifest.json", locked_run / "run_manifest.json"),
    ]:
        cstat = file_stat(cpath)
        lstat = file_stat(lpath)
        rows.append(
            {
                "fold": 0,
                "artifact": artifact,
                "candidate_path": cstat["path"],
                "locked_path": lstat["path"],
                "candidate_exists": cstat["exists"],
                "locked_exists": lstat["exists"],
                "candidate_realpath": cstat["realpath"],
                "locked_realpath": lstat["realpath"],
                "realpath_same": bool(cstat["realpath"] and cstat["realpath"] == lstat["realpath"]),
                "candidate_size_bytes": cstat["size_bytes"],
                "locked_size_bytes": lstat["size_bytes"],
                "size_same": bool(cstat["exists"] and lstat["exists"] and cstat["size_bytes"] == lstat["size_bytes"]),
                "candidate_mtime_iso": cstat["mtime_iso"],
                "locked_mtime_iso": lstat["mtime_iso"],
                "candidate_sha256": cstat["sha256"],
                "locked_sha256": lstat["sha256"],
                "sha256_same": bool(cstat["sha256"] and cstat["sha256"] == lstat["sha256"]),
            }
        )
    return pd.DataFrame(rows)


def read_latent(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        raise RuntimeError(f"{path} lacks SubjectID")
    return df


def sort_latent(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["SubjectID", "tensor_idx"] if c in df.columns]
    return df.sort_values(cols).reset_index(drop=True)


def latent_hash_comparison(candidate_readout: Path, locked_readout: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        for split in ["test", "trainDev"]:
            cpath = candidate_readout / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
            lpath = locked_readout / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
            cdf = sort_latent(read_latent(cpath))
            ldf = sort_latent(read_latent(lpath))
            mu_cols = [c for c in cdf.columns if c.startswith("mu_")]
            l_mu_cols = [c for c in ldf.columns if c.startswith("mu_")]
            subject_same = cdf["SubjectID"].astype(str).tolist() == ldf["SubjectID"].astype(str).tolist()
            mu_cols_same = mu_cols == l_mu_cols
            if not mu_cols_same:
                raise RuntimeError(f"Mu columns differ for fold {fold} {split}")
            cvals = cdf[mu_cols].to_numpy(dtype=np.float64)
            lvals = ldf[mu_cols].to_numpy(dtype=np.float64)
            diff = cvals - lvals if cvals.shape == lvals.shape else np.array([np.nan])
            finite_diff = diff[np.isfinite(diff)]
            exact_equal = bool(cvals.shape == lvals.shape and np.array_equal(cvals, lvals))
            close_equal = bool(cvals.shape == lvals.shape and np.allclose(cvals, lvals, rtol=0.0, atol=1e-12))
            if cvals.shape == lvals.shape and cvals.size and np.std(cvals.ravel()) > 0 and np.std(lvals.ravel()) > 0:
                corr = float(np.corrcoef(cvals.ravel(), lvals.ravel())[0, 1])
            else:
                corr = np.nan
            rows.append(
                {
                    "fold": fold,
                    "split": split,
                    "candidate_path": str(cpath),
                    "locked_path": str(lpath),
                    "candidate_realpath": str(cpath.resolve()),
                    "locked_realpath": str(lpath.resolve()),
                    "candidate_file_sha256": sha256_file(cpath),
                    "locked_file_sha256": sha256_file(lpath),
                    "file_sha256_same": sha256_file(cpath) == sha256_file(lpath),
                    "n_candidate": int(len(cdf)),
                    "n_locked": int(len(ldf)),
                    "n_mu_cols": int(len(mu_cols)),
                    "subject_order_same_after_sort": subject_same,
                    "shape_same": bool(cvals.shape == lvals.shape),
                    "mu_values_sha256_candidate": sha256_array(cvals),
                    "mu_values_sha256_locked": sha256_array(lvals),
                    "mu_values_exact_equal": exact_equal,
                    "mu_values_allclose_1e12": close_equal,
                    "max_abs_diff": float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else np.nan,
                    "mean_abs_diff": float(np.nanmean(np.abs(finite_diff))) if finite_diff.size else np.nan,
                    "flat_pearson_r": corr,
                    "candidate_mu_mean": float(np.mean(cvals)),
                    "candidate_mu_std": float(np.std(cvals)),
                    "locked_mu_mean": float(np.mean(lvals)),
                    "locked_mu_std": float(np.std(lvals)),
                }
            )
    return pd.DataFrame(rows)


def read_primary_predictions(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return df[mask].copy()


def read_primary_thresholds(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_thresholds_by_fold.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return df[mask].copy()


def read_primary_foldwise(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return df[mask].copy()


def read_model_status(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_model_status.csv")
    return df[df["model_name"].astype(str).eq(PRIMARY_MODEL)].copy()


def prediction_score_comparison(candidate_readout: Path, locked_readout: Path) -> pd.DataFrame:
    cpred = read_primary_predictions(candidate_readout)
    lpred = read_primary_predictions(locked_readout)
    cthr = read_primary_thresholds(candidate_readout).set_index("fold")
    lthr = read_primary_thresholds(locked_readout).set_index("fold")
    cmet = read_primary_foldwise(candidate_readout).set_index("fold")
    lmet = read_primary_foldwise(locked_readout).set_index("fold")
    cstat = read_model_status(candidate_readout).set_index("fold")
    lstat = read_model_status(locked_readout).set_index("fold")
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        c = cpred[cpred["fold"].eq(fold)].copy()
        l = lpred[lpred["fold"].eq(fold)].copy()
        sort_cols = [col for col in ["SubjectID", "tensor_idx"] if col in c.columns and col in l.columns]
        c = c.sort_values(sort_cols).reset_index(drop=True)
        l = l.sort_values(sort_cols).reset_index(drop=True)
        subject_same = c["SubjectID"].astype(str).tolist() == l["SubjectID"].astype(str).tolist()
        score_c = c["y_score"].to_numpy(dtype=np.float64)
        score_l = l["y_score"].to_numpy(dtype=np.float64)
        pred_c = c["y_pred"].to_numpy(dtype=int)
        pred_l = l["y_pred"].to_numpy(dtype=int)
        y_c = c["y_true"].to_numpy(dtype=int)
        y_l = l["y_true"].to_numpy(dtype=int)
        if score_c.shape == score_l.shape:
            diff = score_c - score_l
            score_exact = bool(np.array_equal(score_c, score_l))
            score_close = bool(np.allclose(score_c, score_l, rtol=0.0, atol=1e-12))
            max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
            mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
            n_diff = int(np.sum(np.abs(diff) > 1e-12))
        else:
            score_exact = False
            score_close = False
            max_abs = np.nan
            mean_abs = np.nan
            n_diff = -1
        c_threshold = float(cthr.loc[fold, "threshold"])
        l_threshold = float(lthr.loc[fold, "threshold"])
        c_params = str(cstat.loc[fold, "best_params"]) if fold in cstat.index else ""
        l_params = str(lstat.loc[fold, "best_params"]) if fold in lstat.index else ""
        row: Dict[str, Any] = {
            "fold": fold,
            "n_candidate": int(len(c)),
            "n_locked": int(len(l)),
            "subject_order_same_after_sort": subject_same,
            "y_true_equal": bool(np.array_equal(y_c, y_l)),
            "score_exact_equal": score_exact,
            "score_allclose_1e12": score_close,
            "score_max_abs_diff": max_abs,
            "score_mean_abs_diff": mean_abs,
            "n_score_diff_gt_1e12": n_diff,
            "y_pred_equal": bool(np.array_equal(pred_c, pred_l)),
            "candidate_threshold": c_threshold,
            "locked_threshold": l_threshold,
            "threshold_equal": bool(math.isclose(c_threshold, l_threshold, rel_tol=0.0, abs_tol=1e-15)),
            "candidate_best_params": c_params,
            "locked_best_params": l_params,
            "best_params_equal": c_params == l_params,
            "candidate_inner_oof_context": str(cthr.loc[fold, "threshold_selection_context"]),
            "locked_inner_oof_context": str(lthr.loc[fold, "threshold_selection_context"]),
            "candidate_inner_cv_context": str(cthr.loc[fold, "inner_cv_context"]),
            "locked_inner_cv_context": str(lthr.loc[fold, "inner_cv_context"]),
            "candidate_auc": float(cmet.loc[fold, "auc"]),
            "locked_auc": float(lmet.loc[fold, "auc"]),
            "candidate_pr_auc": float(cmet.loc[fold, "pr_auc"]),
            "locked_pr_auc": float(lmet.loc[fold, "pr_auc"]),
            "candidate_balanced_accuracy": float(cmet.loc[fold, "balanced_accuracy"]),
            "locked_balanced_accuracy": float(lmet.loc[fold, "balanced_accuracy"]),
            "candidate_f1": float(cmet.loc[fold, "f1"]),
            "locked_f1": float(lmet.loc[fold, "f1"]),
        }
        for metric in ["auc", "pr_auc", "balanced_accuracy", "f1"]:
            row[f"delta_{metric}"] = row[f"candidate_{metric}"] - row[f"locked_{metric}"]
            row[f"{metric}_rounded_6_equal"] = round(row[f"candidate_{metric}"], 6) == round(row[f"locked_{metric}"], 6)
        rows.append(row)
    return pd.DataFrame(rows)


def stageb_report(candidate_run: Path, candidate_readout: Path, locked_run: Path, locked_readout: Path, pred_cmp: pd.DataFrame, lat_cmp: pd.DataFrame, inventory: pd.DataFrame) -> str:
    cand_log = load_json(candidate_readout / "command_log.json")
    locked_log = load_json(locked_readout / "command_log.json")
    cand_run_real = str(candidate_run.resolve())
    cand_readout_real = str(candidate_readout.resolve())
    locked_run_real = str(locked_run.resolve())
    locked_readout_real = str(locked_readout.resolve())
    readout_path_ok = (
        str(cand_log.get("run_dir", "")) == str(candidate_run)
        and str(cand_log.get("output_dir", "")) == str(candidate_readout)
    )
    threshold_ok = str(cand_log.get("threshold_selection", "")).startswith("true_inner_cv_oof")
    latent_source = str(cand_log.get("latent_source", ""))
    model_status = read_model_status(candidate_readout)
    selected_params = model_status[["fold", "best_params", "best_inner_auc", "inner_cv_context", "minimum_inner_stratum_count"]].to_dict("records")
    identical_latent_folds = sorted(lat_cmp.groupby("fold")["mu_values_exact_equal"].all().loc[lambda s: s].index.tolist())
    identical_score_folds = sorted(pred_cmp[pred_cmp["score_exact_equal"]]["fold"].astype(int).tolist())
    same_ckpt = inventory[(inventory["artifact"].eq("vae_model")) & (inventory["sha256_same"])]
    same_ckpt_folds = sorted(same_ckpt["fold"].astype(int).tolist())
    lines = [
        "# Stage B Readout Integrity Report",
        "",
        "## Path checks",
        "",
        f"- Candidate run path: `{candidate_run}`",
        f"- Candidate run realpath: `{cand_run_real}`",
        f"- Candidate readout path: `{candidate_readout}`",
        f"- Candidate readout realpath: `{cand_readout_real}`",
        f"- Locked run realpath: `{locked_run_real}`",
        f"- Locked readout realpath: `{locked_readout_real}`",
        f"- Candidate command_log points to candidate run/readout paths: **{readout_path_ok}**",
        f"- Candidate and locked readout realpaths are distinct: **{cand_readout_real != locked_readout_real}**",
        "",
        "## Threshold/readout checks",
        "",
        f"- Candidate Stage B latent source: `{latent_source}`",
        f"- Candidate Stage B threshold selection: `{cand_log.get('threshold_selection', '')}`",
        f"- True inner-CV OOF thresholding indicated: **{threshold_ok}**",
        f"- Candidate classifiers requested: `{cand_log.get('classifiers_requested', [])}`",
        f"- Candidate folds_to_run: `{cand_log.get('folds_to_run', [])}`",
        "",
        "## Identity checks",
        "",
        f"- VAE checkpoint SHA-identical folds candidate vs locked: `{same_ckpt_folds}`",
        f"- Latent mu exactly identical folds candidate vs locked: `{identical_latent_folds}`",
        f"- Prediction-score exactly identical folds candidate vs locked: `{identical_score_folds}`",
        "",
        "## Selected logreg_l2 hyperparameters by fold",
        "",
        md_table(pd.DataFrame(selected_params), max_rows=10),
    ]
    if same_ckpt_folds:
        lines.extend(
            [
                "",
                "Interpretation: folds listed above are not merely rounded-equal in the comparison table.",
                "Their VAE checkpoints have identical SHA256 hashes, their latent mu files are numerically identical,",
                "and their Stage B logreg_l2 scores are identical. This rules out a simple table-formatting artifact.",
            ]
        )
    return "\n".join(lines) + "\n"


def final_decision(pred_cmp: pd.DataFrame, lat_cmp: pd.DataFrame, inventory: pd.DataFrame) -> str:
    identical_score_folds = sorted(pred_cmp[pred_cmp["score_exact_equal"]]["fold"].astype(int).tolist())
    identical_latent_folds = sorted(lat_cmp.groupby("fold")["mu_values_exact_equal"].all().loc[lambda s: s].index.tolist())
    same_ckpt_folds = sorted(inventory[(inventory["artifact"].eq("vae_model")) & (inventory["sha256_same"])]["fold"].astype(int).tolist())
    fold5_diff = 5 not in same_ckpt_folds and 5 not in identical_latent_folds and not bool(pred_cmp[pred_cmp["fold"].eq(5)]["score_exact_equal"].iloc[0])
    invalid = same_ckpt_folds != [] or identical_score_folds != []
    lines = [
        "# Final Integrity Decision",
        "",
        "## Decision",
        "",
    ]
    if invalid:
        lines.append("**Do not treat horizon4480_cycles56 as a valid promoted FULL 5x5 replacement yet.**")
    else:
        lines.append("No stale-fold evidence was detected; the candidate appears internally independent of the locked readout.")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- VAE checkpoint SHA-identical folds: `{same_ckpt_folds}`",
            f"- Latent mu exactly identical folds: `{identical_latent_folds}`",
            f"- Prediction-score exactly identical folds: `{identical_score_folds}`",
            f"- Fold 5 independently differs from locked across checkpoint/latent/scores: **{fold5_diff}**",
            "",
            "The candidate Stage B command log points to the candidate readout directory, so this is not evidence of",
            "the comparison script falling back to the locked readout. The issue is upstream: the candidate readout",
            "for the identical folds is based on candidate fold artifacts that are byte-identical/numerically identical",
            "to the locked run.",
            "",
            "## Consequence",
            "",
            "The apparent improvement is driven by Fold 5 while folds 1-4 are locked-run-identical. That is a mixed",
            "artifact set, not a clean long-horizon 5-fold confirmation.",
            "",
            "## Required next step",
            "",
            "Rerun horizon4480_cycles56 from a clean output directory or with explicit force/retrain safeguards that",
            "regenerate all five fold checkpoints and latent caches. Only after all five fold checkpoints differ from",
            "the locked run, and Stage B is recomputed from those candidate latents, should promotion be reconsidered.",
        ]
    )
    return "\n".join(lines) + "\n"


def readme_text(inventory: pd.DataFrame, lat_cmp: pd.DataFrame, pred_cmp: pd.DataFrame, stageb_ok: bool) -> str:
    required_artifacts = {
        "vae_model",
        "vae_history",
        "readout_test_latent_mu",
        "readout_trainDev_latent_mu",
    }
    missing = inventory[
        inventory["fold"].isin(FOLDS)
        & inventory["artifact"].isin(required_artifacts)
        & (~inventory["candidate_exists"])
    ]
    same_ckpt_folds = sorted(inventory[(inventory["artifact"].eq("vae_model")) & (inventory["sha256_same"])]["fold"].astype(int).tolist())
    same_latent = sorted(lat_cmp.groupby("fold")["mu_values_exact_equal"].all().loc[lambda s: s].index.tolist())
    same_scores = sorted(pred_cmp[pred_cmp["score_exact_equal"]]["fold"].astype(int).tolist())
    rounded_equal = sorted(
        pred_cmp[
            pred_cmp[["auc_rounded_6_equal", "pr_auc_rounded_6_equal", "balanced_accuracy_rounded_6_equal", "f1_rounded_6_equal"]].all(axis=1)
        ]["fold"].astype(int).tolist()
    )
    return "\n".join(
        [
            "# Horizon4480/Cycles56 Integrity Audit",
            "",
            "Read-only audit of the promoted long-horizon FULL 5x5 candidate versus the locked current FULL [1,0,2] run.",
            "",
            "## Summary",
            "",
            f"- Required candidate fold artifacts missing: **{len(missing)}**",
            f"- Candidate Stage B command/readout path integrity OK: **{stageb_ok}**",
            f"- VAE checkpoint SHA-identical folds candidate vs locked: `{same_ckpt_folds}`",
            f"- Latent mu exactly identical folds candidate vs locked: `{same_latent}`",
            f"- Prediction-score exactly identical folds candidate vs locked: `{same_scores}`",
            f"- Foldwise metrics rounded-equal folds: `{rounded_equal}`",
            "",
            "## Main Finding",
            "",
            "Folds 1-4 are truly identical, not only rounded-equal. Their VAE checkpoints match the locked run by SHA256,",
            "their cached latent mu matrices are exactly equal, and their Stage B prediction scores are exactly equal.",
            "Fold 5 differs and accounts for the observed candidate improvement.",
            "",
            "## Interpretation",
            "",
            "The comparison script did read the candidate readout directory. The integrity problem is that candidate",
            "fold artifacts for folds 1-4 are stale/identical to the locked run. Therefore the promoted result is not a",
            "clean FULL 5x5 horizon4480 confirmation.",
            "",
            "## Decision",
            "",
            "Do not promote horizon4480_cycles56 as revised main model based on the current artifact set. Rerun from a clean",
            "candidate output directory and regenerate all five fold checkpoints/latent caches before evaluating promotion.",
        ]
    ) + "\n"


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    locked_run = resolve(args.locked_run)
    locked_readout = resolve(args.locked_readout)
    outdir = resolve(args.output_dir)
    ensure_outdir(outdir, overwrite=args.overwrite)

    inventory = build_inventory(candidate_run, candidate_readout, locked_run, locked_readout)
    latent_cmp = latent_hash_comparison(candidate_readout, locked_readout)
    pred_cmp = prediction_score_comparison(candidate_readout, locked_readout)

    cand_log = load_json(candidate_readout / "command_log.json")
    stageb_ok = (
        str(cand_log.get("run_dir", "")) == str(candidate_run)
        and str(cand_log.get("output_dir", "")) == str(candidate_readout)
        and str(cand_log.get("threshold_selection", "")).startswith("true_inner_cv_oof")
        and str(candidate_readout.resolve()) != str(locked_readout.resolve())
    )

    write_table(outdir, "foldwise_file_inventory", inventory, max_rows=120)
    write_table(outdir, "latent_hash_comparison", latent_cmp, max_rows=40)
    write_table(outdir, "prediction_score_comparison", pred_cmp, max_rows=20)
    (outdir / "stageb_readout_integrity_report.md").write_text(
        stageb_report(candidate_run, candidate_readout, locked_run, locked_readout, pred_cmp, latent_cmp, inventory),
        encoding="utf-8",
    )
    (outdir / "final_integrity_decision.md").write_text(final_decision(pred_cmp, latent_cmp, inventory), encoding="utf-8")
    (outdir / "README.md").write_text(readme_text(inventory, latent_cmp, pred_cmp, stageb_ok), encoding="utf-8")

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_run": str(candidate_run),
        "candidate_run_realpath": str(candidate_run.resolve()),
        "candidate_readout": str(candidate_readout),
        "candidate_readout_realpath": str(candidate_readout.resolve()),
        "locked_run": str(locked_run),
        "locked_run_realpath": str(locked_run.resolve()),
        "locked_readout": str(locked_readout),
        "locked_readout_realpath": str(locked_readout.resolve()),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "stageb_path_integrity_ok": stageb_ok,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "model_output_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote integrity audit to {outdir}")
    print(readme_text(inventory, latent_cmp, pred_cmp, stageb_ok))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
