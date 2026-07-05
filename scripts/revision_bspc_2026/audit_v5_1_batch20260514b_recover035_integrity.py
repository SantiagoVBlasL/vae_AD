#!/usr/bin/env python3
"""Read-only integrity audit for the recover035 FULL 5x5 metadata-rescue retrain.

Pre-training (--dry-run):
  - Verifies zero parameter diffs vs locked v5.1b.
  - Verifies metadata_path points to patched_metadata_candidate.csv.
  - Verifies global_tensor_path unchanged.
  - Lists what would be checked post-training.

Post-training:
  - Verifies all candidate fold artifacts exist.
  - Verifies candidate VAE checkpoints are NOT SHA-identical to locked (training used +1 subject).
  - Verifies 035_S_6927 appears in candidate latent caches for its test fold and trainDev for others.
  - Verifies 128_S_2002 is absent from all candidate latent caches.
  - Full Stage B readout integrity check.
  - Does not train, score, or modify any existing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "recover035_full5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
LOCKED_READOUT = LOCKED_RUN / "classifier_only_readout"
OUTPUT_DIR = RESULTS / "recover035_full5x5_integrity_audit"

SOURCE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
)
CANDIDATE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_recover035_full5x5.json"
)
PATCHED_METADATA_SUFFIX = "patched_metadata_candidate.csv"
ORIGINAL_METADATA_SUFFIX = "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Config-level checks only; no artifact files are read and no output files written.",
    )
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
        "path": str(path), "exists": bool(exists), "realpath": str(path.resolve()) if exists else "",
        "size_bytes": np.nan, "mtime_epoch": np.nan, "mtime_iso": "", "sha256": "",
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


def dry_run_config_check() -> None:
    """Config-level verification only; prints analysis and exits 0."""
    errors: List[str] = []
    src_cfg = load_json(SOURCE_CONFIG_PATH)
    cand_cfg = load_json(CANDIDATE_CONFIG_PATH)
    if not src_cfg:
        errors.append(f"Source config not found: {SOURCE_CONFIG_PATH}")
    if not cand_cfg:
        errors.append(f"Candidate config not found: {CANDIDATE_CONFIG_PATH}")
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        raise SystemExit(1)

    src_params = dict(src_cfg.get("parameters", {}))
    tgt_params = dict(cand_cfg.get("parameters", {}))
    for params in (src_params, tgt_params):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
    param_diffs = {k: (src_params.get(k), tgt_params.get(k)) for k in set(list(src_params) + list(tgt_params)) if src_params.get(k) != tgt_params.get(k)}

    src_meta = src_cfg.get("paths", {}).get("metadata_path", "")
    tgt_meta = cand_cfg.get("paths", {}).get("metadata_path", "")
    src_tensor = src_cfg.get("paths", {}).get("global_tensor_path", "")
    tgt_tensor = cand_cfg.get("paths", {}).get("global_tensor_path", "")

    print("# recover035 Integrity Audit — Dry-Run Report")
    print()
    print("## Config-level verification")
    print()
    print(f"- Parameter diffs vs locked source    : {len(param_diffs)} (expected 0)")
    for k, (sv, tv) in sorted(param_diffs.items()):
        print(f"    {k}: {sv!r} -> {tv!r}  [UNEXPECTED]")
    if not param_diffs:
        print("  -> zero parameter diffs  [VERIFIED]")
    print(f"- metadata_path changed               : {src_meta != tgt_meta}  [{'VERIFIED' if src_meta != tgt_meta else 'FAIL'}]")
    print(f"  locked:    {src_meta}")
    print(f"  candidate: {tgt_meta}")
    print(f"  ends with patched_metadata_candidate.csv: {tgt_meta.endswith(PATCHED_METADATA_SUFFIX)}  [{'VERIFIED' if tgt_meta.endswith(PATCHED_METADATA_SUFFIX) else 'FAIL'}]")
    print(f"  original metadata NOT used           : {ORIGINAL_METADATA_SUFFIX not in tgt_meta}  [{'VERIFIED' if ORIGINAL_METADATA_SUFFIX not in tgt_meta else 'FAIL'}]")
    print(f"- global_tensor_path unchanged         : {src_tensor == tgt_tensor}  [{'VERIFIED' if src_tensor == tgt_tensor else 'FAIL'}]")
    print()
    print("## What would be checked post-training")
    print()
    print("- VAE checkpoint SHA256: candidate must DIFFER from locked (retrained on +1 subject)")
    print("- Latent mu: 035_S_6927 must appear in test latent cache for its outer-test fold")
    print("- Latent mu: 035_S_6927 must appear in trainDev latent cache for all other folds")
    print(f"- Latent mu: {EXCLUDED_SUBJECT} must be absent from ALL latent caches")
    print("- Stage B predictions: 035_S_6927 prediction must appear for its outer-test fold")
    print("- Stage B command_log: must point to candidate run/readout paths")
    print("- Stage B threshold: true_inner_cv_oof strategy confirmed")
    print("- Foldwise metrics: AUC, PR-AUC, BA, Sens, Spec, F1 per fold")
    print()
    print("## VAE internal val health checks (new — prevents invalid partial-run acceptance)")
    print()
    print("- Per fold: vae_internal_val_idx_local_to_pool.npy must exist and have len > 0")
    print(f"  (len == 0 means early stopping was silently disabled — run is NOT comparable to locked v5.1b)")
    print(f"- Per fold: vae_pool_required_metadata_removed_fold_N.csv must exist and contain {EXCLUDED_SUBJECT}")
    print(f"  (confirms --vae_required_metadata_cols was active and excluded the tensor-only subject)")
    print()
    print(f"Candidate run  : {CANDIDATE_RUN}")
    print(f"Locked run     : {LOCKED_RUN}")
    print()
    print("Dry-run complete. No audit output files were written.")


def ensure_outdir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already contains files; pass --overwrite")
    if overwrite:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
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
        "stagea_test_predictions_logreg": fold_dir / "test_predictions_logreg.csv",
        "stagea_test_predictions_svm": fold_dir / "test_predictions_svm.csv",
        "readout_test_latent_mu": readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        "readout_trainDev_latent_mu": readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
    }


def build_inventory(
    candidate_run: Path, candidate_readout: Path, locked_run: Path, locked_readout: Path
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        cand = artifact_paths(candidate_run, candidate_readout, fold)
        lock = artifact_paths(locked_run, locked_readout, fold)
        for artifact in sorted(cand):
            cstat = file_stat(cand[artifact])
            lstat = file_stat(lock[artifact])
            rows.append({
                "fold": fold, "artifact": artifact,
                "candidate_path": cstat["path"], "locked_path": lstat["path"],
                "candidate_exists": cstat["exists"], "locked_exists": lstat["exists"],
                "realpath_same": bool(cstat["realpath"] and cstat["realpath"] == lstat["realpath"]),
                "candidate_size_bytes": cstat["size_bytes"], "locked_size_bytes": lstat["size_bytes"],
                "size_same": bool(cstat["exists"] and lstat["exists"] and cstat["size_bytes"] == lstat["size_bytes"]),
                "candidate_mtime_iso": cstat["mtime_iso"], "locked_mtime_iso": lstat["mtime_iso"],
                "candidate_sha256": cstat["sha256"], "locked_sha256": lstat["sha256"],
                "sha256_same": bool(cstat["sha256"] and cstat["sha256"] == lstat["sha256"]),
            })
    for artifact, cpath, lpath in [
        ("stageb_predictions", candidate_readout / "classifier_sweep_predictions.csv", locked_readout / "classifier_sweep_predictions.csv"),
        ("stageb_foldwise_metrics", candidate_readout / "classifier_sweep_foldwise_metrics.csv", locked_readout / "classifier_sweep_foldwise_metrics.csv"),
        ("stageb_command_log", candidate_readout / "command_log.json", locked_readout / "command_log.json"),
        ("stagea_run_manifest", candidate_run / "run_manifest.json", locked_run / "run_manifest.json"),
    ]:
        cstat = file_stat(cpath)
        lstat = file_stat(lpath)
        rows.append({
            "fold": 0, "artifact": artifact,
            "candidate_path": cstat["path"], "locked_path": lstat["path"],
            "candidate_exists": cstat["exists"], "locked_exists": lstat["exists"],
            "realpath_same": bool(cstat["realpath"] and cstat["realpath"] == lstat["realpath"]),
            "candidate_size_bytes": cstat["size_bytes"], "locked_size_bytes": lstat["size_bytes"],
            "size_same": bool(cstat["exists"] and lstat["exists"] and cstat["size_bytes"] == lstat["size_bytes"]),
            "candidate_mtime_iso": cstat["mtime_iso"], "locked_mtime_iso": lstat["mtime_iso"],
            "candidate_sha256": cstat["sha256"], "locked_sha256": lstat["sha256"],
            "sha256_same": bool(cstat["sha256"] and cstat["sha256"] == lstat["sha256"]),
        })
    return pd.DataFrame(rows)


def check_recover_subject_in_latents(candidate_readout: Path) -> pd.DataFrame:
    """Report 035_S_6927 and 128_S_2002 presence in each fold's latent caches."""
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        for split in ["test", "trainDev"]:
            path = candidate_readout / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
            if not path.exists():
                rows.append({
                    "fold": fold, "split": split, "file_exists": False,
                    f"{RECOVER_SUBJECT}_present": None,
                    f"{EXCLUDED_SUBJECT}_present": None,
                    "note": "file not found",
                })
                continue
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            if "SubjectID" not in df.columns:
                rows.append({
                    "fold": fold, "split": split, "file_exists": True,
                    f"{RECOVER_SUBJECT}_present": None,
                    f"{EXCLUDED_SUBJECT}_present": None,
                    "note": "SubjectID column missing",
                })
                continue
            subjects = set(df["SubjectID"].astype(str).tolist())
            rows.append({
                "fold": fold, "split": split, "file_exists": True,
                f"{RECOVER_SUBJECT}_present": RECOVER_SUBJECT in subjects,
                f"{EXCLUDED_SUBJECT}_present": EXCLUDED_SUBJECT in subjects,
                "n_subjects": int(len(df)),
                "note": "",
            })
    return pd.DataFrame(rows)


def check_vae_val_health(candidate_run: Path) -> pd.DataFrame:
    """Per-fold check that VAE internal val was non-empty and 128_S_2002 was excluded.

    Checks:
    - vae_internal_val_idx_local_to_pool.npy exists and has len > 0
    - vae_pool_required_metadata_removed_fold_N.csv exists and contains 128_S_2002
    """
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fold_dir = candidate_run / f"fold_{fold}"
        val_idx_path = fold_dir / "vae_internal_val_idx_local_to_pool.npy"
        removed_csv_path = fold_dir / f"vae_pool_required_metadata_removed_fold_{fold}.csv"

        val_idx_exists = val_idx_path.exists()
        val_n: Any = None
        val_nonempty: Any = None
        if val_idx_exists:
            try:
                val_idx = np.load(val_idx_path)
                val_n = int(len(val_idx))
                val_nonempty = val_n > 0
            except Exception:
                val_nonempty = None

        removed_exists = removed_csv_path.exists()
        excluded_removed: Any = None
        if removed_exists:
            try:
                removed_df = pd.read_csv(removed_csv_path, dtype=str)
                excluded_removed = bool((removed_df["SubjectID"].astype(str) == EXCLUDED_SUBJECT).any())
            except Exception:
                excluded_removed = False

        all_ok = bool(val_idx_exists and val_nonempty and removed_exists and excluded_removed)
        rows.append({
            "fold": fold,
            "vae_val_idx_file_exists": val_idx_exists,
            "vae_internal_val_n": val_n,
            "vae_internal_val_nonempty": val_nonempty,
            "vae_required_metadata_removed_csv_exists": removed_exists,
            f"{EXCLUDED_SUBJECT}_confirmed_removed": excluded_removed,
            "all_checks_ok": all_ok,
        })
    return pd.DataFrame(rows)


def read_primary_predictions(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    return df[mask].copy()


def read_primary_foldwise(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_foldwise_metrics.csv")
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    return df[mask].copy()


def read_primary_thresholds(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_thresholds_by_fold.csv")
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    return df[mask].copy()


def read_model_status(readout: Path) -> pd.DataFrame:
    p = readout / "classifier_sweep_model_status.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["model_name"].astype(str).eq(PRIMARY_MODEL)].copy()


def prediction_score_comparison(candidate_readout: Path, locked_readout: Path) -> pd.DataFrame:
    cpred = read_primary_predictions(candidate_readout)
    lpred = read_primary_predictions(locked_readout)
    cmet = read_primary_foldwise(candidate_readout).set_index("fold")
    lmet = read_primary_foldwise(locked_readout).set_index("fold")
    cthr = read_primary_thresholds(candidate_readout).set_index("fold")
    lthr = read_primary_thresholds(locked_readout).set_index("fold")
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        c = cpred[cpred["fold"].eq(fold)].copy()
        l = lpred[lpred["fold"].eq(fold)].copy()
        sort_cols = [col for col in ["SubjectID", "tensor_idx"] if col in c.columns and col in l.columns]
        c = c.sort_values(sort_cols).reset_index(drop=True)
        l = l.sort_values(sort_cols).reset_index(drop=True)
        subject_same = c["SubjectID"].astype(str).tolist() == l["SubjectID"].astype(str).tolist()
        # Note: subjects differ because recover035 has one extra test subject in one fold
        score_c = c["y_score"].to_numpy(dtype=np.float64) if not c.empty else np.array([])
        score_l = l["y_score"].to_numpy(dtype=np.float64) if not l.empty else np.array([])
        c_threshold = float(cthr.loc[fold, "threshold"]) if fold in cthr.index else np.nan
        l_threshold = float(lthr.loc[fold, "threshold"]) if fold in lthr.index else np.nan
        recover_in_test = RECOVER_SUBJECT in set(c["SubjectID"].astype(str).tolist()) if "SubjectID" in c.columns else None
        row: Dict[str, Any] = {
            "fold": fold,
            "n_candidate": int(len(c)),
            "n_locked": int(len(l)),
            f"{RECOVER_SUBJECT}_in_test": recover_in_test,
            "subject_sets_same": subject_same,
            "candidate_threshold": c_threshold,
            "locked_threshold": l_threshold,
            "threshold_equal": bool(math.isclose(c_threshold, l_threshold, rel_tol=0.0, abs_tol=1e-15)) if not (math.isnan(c_threshold) or math.isnan(l_threshold)) else False,
            "candidate_auc": float(cmet.loc[fold, "auc"]) if fold in cmet.index else np.nan,
            "locked_auc": float(lmet.loc[fold, "auc"]) if fold in lmet.index else np.nan,
            "candidate_pr_auc": float(cmet.loc[fold, "pr_auc"]) if fold in cmet.index else np.nan,
            "locked_pr_auc": float(lmet.loc[fold, "pr_auc"]) if fold in lmet.index else np.nan,
            "candidate_balanced_accuracy": float(cmet.loc[fold, "balanced_accuracy"]) if fold in cmet.index else np.nan,
            "locked_balanced_accuracy": float(lmet.loc[fold, "balanced_accuracy"]) if fold in lmet.index else np.nan,
        }
        for metric in ["auc", "pr_auc", "balanced_accuracy"]:
            c_val = row[f"candidate_{metric}"]
            l_val = row[f"locked_{metric}"]
            row[f"delta_{metric}"] = c_val - l_val if not (math.isnan(c_val) or math.isnan(l_val)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def stageb_report_text(
    candidate_run: Path, candidate_readout: Path, locked_run: Path, locked_readout: Path,
    inventory: pd.DataFrame, subject_check: pd.DataFrame, pred_cmp: pd.DataFrame,
    vae_val_health: pd.DataFrame,
) -> str:
    cand_log = load_json(candidate_readout / "command_log.json")
    cand_run_real = str(candidate_run.resolve())
    cand_readout_real = str(candidate_readout.resolve())
    locked_readout_real = str(locked_readout.resolve())
    readout_path_ok = (
        str(cand_log.get("run_dir", "")) == str(candidate_run)
        and str(cand_log.get("output_dir", "")) == str(candidate_readout)
    )
    threshold_ok = str(cand_log.get("threshold_selection", "")).startswith("true_inner_cv_oof")

    # Checkpoint identity check: all candidate checkpoints must DIFFER from locked
    ckpt_same = inventory[(inventory["artifact"].eq("vae_model")) & (inventory["sha256_same"])]
    ckpt_same_folds = sorted(ckpt_same["fold"].astype(int).tolist())
    ckpt_different_folds = sorted(set(FOLDS) - set(ckpt_same_folds))
    ckpt_ok = len(ckpt_same_folds) == 0

    # 035 in latents
    col_035 = f"{RECOVER_SUBJECT}_present"
    col_128 = f"{EXCLUDED_SUBJECT}_present"
    test_035 = subject_check[(subject_check["split"] == "test") & subject_check.get(col_035, pd.Series(dtype=bool)).fillna(False)]
    test_035_folds = sorted(test_035["fold"].astype(int).tolist()) if not test_035.empty else []
    train_035_missing = subject_check[
        (subject_check["split"] == "trainDev")
        & subject_check.get(col_035, pd.Series(dtype=bool)).fillna(True).map(lambda x: x is False)
    ] if col_035 in subject_check.columns else pd.DataFrame()
    excluded_present = subject_check[subject_check.get(col_128, pd.Series(dtype=bool)).fillna(False)] if col_128 in subject_check.columns else pd.DataFrame()

    recover_test_fold = None
    if f"{RECOVER_SUBJECT}_in_test" in pred_cmp.columns:
        hit = pred_cmp[pred_cmp[f"{RECOVER_SUBJECT}_in_test"] == True]
        if not hit.empty:
            recover_test_fold = int(hit["fold"].iloc[0])

    model_status = read_model_status(candidate_readout)
    lines = [
        "# Stage B Readout Integrity Report (recover035)",
        "",
        "## Path checks",
        f"- Candidate run path: `{candidate_run}`",
        f"- Candidate readout path: `{candidate_readout}`",
        f"- Candidate command_log points to candidate run/readout: **{readout_path_ok}**",
        f"- Candidate and locked realpaths are distinct: **{cand_readout_real != locked_readout_real}**",
        "",
        "## Threshold/readout checks",
        f"- Stage B threshold selection: `{cand_log.get('threshold_selection', '')}`",
        f"- True inner-CV OOF thresholding: **{threshold_ok}**",
        "",
        "## VAE checkpoint freshness (must differ from locked — different training data)",
        f"- Checkpoints DIFFERENT from locked v5.1b: folds `{ckpt_different_folds}`",
        f"- Checkpoints SHA-identical to locked (unexpected): folds `{ckpt_same_folds}`",
        f"- All checkpoints differ from locked: **{ckpt_ok}**",
        "",
        "## Rescue subject 035_S_6927 in latent caches",
        f"- 035_S_6927 test-latent folds: `{test_035_folds}`",
        f"- 035_S_6927 outer-test fold (from predictions): `{recover_test_fold}`",
        f"- 035_S_6927 absent from trainDev where expected: {len(train_035_missing) == 0}",
        "",
        "## Excluded subject 128_S_2002 in latent caches",
        f"- 128_S_2002 appears in any latent cache (should be 0): **{not excluded_present.empty}**",
        "",
        "## VAE internal val health (required for valid early stopping)",
        "",
    ]
    if not vae_val_health.empty:
        val_all_ok = bool(vae_val_health["all_checks_ok"].all())
        lines.append(f"- All folds pass VAE val health checks: **{val_all_ok}**")
        bad_val = vae_val_health[~vae_val_health["vae_internal_val_nonempty"].fillna(False)]
        if not bad_val.empty:
            lines.append(f"- Folds with empty val set (early stopping disabled): `{sorted(bad_val['fold'].tolist())}`")
        bad_excl = vae_val_health[~vae_val_health[f"{EXCLUDED_SUBJECT}_confirmed_removed"].fillna(False)]
        if not bad_excl.empty:
            lines.append(f"- Folds where {EXCLUDED_SUBJECT} removal not confirmed: `{sorted(bad_excl['fold'].tolist())}`")
        lines.append("")
        lines.append(md_table(vae_val_health, max_rows=10))
    else:
        lines.append("_(vae_val_health not available)_")
    lines.append("")
    if not model_status.empty:
        lines.append("## Selected logreg_l2 hyperparameters by fold")
        lines.append("")
        lines.append(md_table(model_status[["fold", "best_params", "best_inner_auc"]].head(5), max_rows=5))
    return "\n".join(lines) + "\n"


def final_decision_text(
    inventory: pd.DataFrame,
    subject_check: pd.DataFrame,
    pred_cmp: pd.DataFrame,
    vae_val_health: pd.DataFrame,
) -> str:
    ckpt_same = inventory[(inventory["artifact"].eq("vae_model")) & (inventory["sha256_same"])]
    ckpt_same_folds = sorted(ckpt_same["fold"].astype(int).tolist())
    col_128 = f"{EXCLUDED_SUBJECT}_present"
    excluded_present = subject_check[subject_check.get(col_128, pd.Series(dtype=bool)).fillna(False)] if col_128 in subject_check.columns else pd.DataFrame()
    invalid_conditions = []
    if ckpt_same_folds:
        invalid_conditions.append(f"VAE checkpoints SHA-identical to locked for folds {ckpt_same_folds} — retrain may not have run cleanly")
    if not excluded_present.empty:
        invalid_conditions.append(f"{EXCLUDED_SUBJECT} appears in candidate latent caches — should be absent")
    if not vae_val_health.empty:
        bad_val = vae_val_health[~vae_val_health["vae_internal_val_nonempty"].fillna(False)]
        if not bad_val.empty:
            invalid_conditions.append(
                f"VAE internal val is empty for folds {sorted(bad_val['fold'].tolist())} — "
                "early stopping was disabled; run is NOT comparable to locked v5.1b"
            )
        bad_excl = vae_val_health[~vae_val_health[f"{EXCLUDED_SUBJECT}_confirmed_removed"].fillna(False)]
        if not bad_excl.empty:
            invalid_conditions.append(
                f"{EXCLUDED_SUBJECT} removal from VAE pool not confirmed for folds {sorted(bad_excl['fold'].tolist())}"
            )
    lines = [
        "# Final Integrity Decision (recover035)",
        "",
        "## Decision",
        "",
    ]
    if invalid_conditions:
        lines.append("**Integrity issues detected. Do not treat recover035 as a valid retrain.**")
        lines.append("")
        for c in invalid_conditions:
            lines.append(f"- {c}")
    else:
        lines.append(
            "No integrity issues detected. Candidate checkpoints differ from locked (as expected for a new training run "
            "with +1 subject), 128_S_2002 is absent from all latent caches, and VAE internal val was non-empty in all folds."
        )
    val_all_ok = bool(vae_val_health["all_checks_ok"].all()) if not vae_val_health.empty else None
    lines.extend([
        "",
        "## Evidence",
        f"- VAE checkpoint SHA-identical folds candidate vs locked: `{ckpt_same_folds}` (expected empty)",
        f"- {EXCLUDED_SUBJECT} in any latent cache: `{not excluded_present.empty}` (expected False)",
        f"- VAE internal val non-empty all folds: `{val_all_ok}` (expected True)",
        f"- {EXCLUDED_SUBJECT} confirmed removed from VAE pool all folds: "
        f"`{bool(vae_val_health[f'{EXCLUDED_SUBJECT}_confirmed_removed'].all()) if not vae_val_health.empty else None}` (expected True)",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    if args.dry_run:
        dry_run_config_check()
        return 0

    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    locked_run = resolve(args.locked_run)
    locked_readout = resolve(args.locked_readout)
    outdir = resolve(args.output_dir)
    ensure_outdir(outdir, overwrite=args.overwrite)

    inventory = build_inventory(candidate_run, candidate_readout, locked_run, locked_readout)
    subject_check = check_recover_subject_in_latents(candidate_readout)
    pred_cmp = prediction_score_comparison(candidate_readout, locked_readout)
    vae_val_health = check_vae_val_health(candidate_run)
    cand_log = load_json(candidate_readout / "command_log.json")
    stageb_ok = (
        str(cand_log.get("run_dir", "")) == str(candidate_run)
        and str(cand_log.get("output_dir", "")) == str(candidate_readout)
        and str(cand_log.get("threshold_selection", "")).startswith("true_inner_cv_oof")
        and str(candidate_readout.resolve()) != str(locked_readout.resolve())
    )

    write_table(outdir, "foldwise_file_inventory", inventory, max_rows=120)
    write_table(outdir, "recover_subject_latent_check", subject_check, max_rows=20)
    write_table(outdir, "prediction_score_comparison", pred_cmp, max_rows=20)
    write_table(outdir, "vae_val_health", vae_val_health, max_rows=10)
    (outdir / "stageb_readout_integrity_report.md").write_text(
        stageb_report_text(
            candidate_run, candidate_readout, locked_run, locked_readout,
            inventory, subject_check, pred_cmp, vae_val_health,
        ),
        encoding="utf-8",
    )
    (outdir / "final_integrity_decision.md").write_text(
        final_decision_text(inventory, subject_check, pred_cmp, vae_val_health),
        encoding="utf-8",
    )

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_run": str(candidate_run),
        "candidate_run_realpath": str(candidate_run.resolve()),
        "candidate_readout": str(candidate_readout),
        "candidate_readout_realpath": str(candidate_readout.resolve()),
        "locked_run": str(locked_run),
        "locked_readout": str(locked_readout),
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "stageb_path_integrity_ok": stageb_ok,
        "vae_val_health_all_folds_ok": bool(vae_val_health["all_checks_ok"].all()) if not vae_val_health.empty else None,
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote integrity audit to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
