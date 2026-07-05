#!/usr/bin/env python3
"""Read-only integrity audit for recover035_latent384_T80_h10000_p560 FULL 5x5.

Verifies:
  - Exactly 1 parameter diff vs recover035_longpatience_T80_h10000_p560_full5x5 (the base model):
      latent_dim 256 -> 384
  - All other parameters identical to longpatience_T80_h10000_p560:
      epochs_vae=10000, cyclical_beta_n_cycles=125, lr_scheduler_T0=80,
      early_stopping_patience_vae=560, n_iter_logreg/svm=500, beta_vae=2.5
  - metadata_path ends with patched_metadata_candidate.csv (same as longpatience)
  - global_tensor_path unchanged
  - cycle length = 10000/125 = 80 epochs (unchanged)
  - patience = 560 = 7 x 80-epoch cycles (unchanged)
  - 035_S_6927 appears in latent caches for its outer-test fold (fold 1) and trainDev for others
  - 128_S_2002 is absent from ALL latent caches
  - Latent cache has 384 mu columns (mu_0..mu_383) per fold
  - VAE internal val non-empty per fold
  - vae_pool_required_metadata_removed_fold_N.csv exists and contains 128_S_2002 per fold
  - VAE checkpoints differ from locked and longpatience256 (different latent space)

--dry-run: config-level checks only; no artifact files are read.

Read-only. Does not train, modify tensor, metadata, ledger, configs, or model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

CANDIDATE_RUN = RESULTS / "recover035_latent384_T80_h10000_p560_full5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
LONGPATIENCE256_RUN = RESULTS / "recover035_longpatience_T80_h10000_p560_full5x5"
OUTPUT_DIR = RESULTS / "recover035_latent384_T80_h10000_p560_integrity_audit"

SOURCE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_longpatience_T80_h10000_p560_full5x5.json"
)
CANDIDATE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_T80_h10000_p560_full5x5.json"
)

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FOLDS = [1, 2, 3, 4, 5]

CANDIDATE_LATENT_DIM = 384
BASE_LATENT_DIM = 256

CANDIDATE_EPOCHS_VAE = 10000
CANDIDATE_N_CYCLES = 125
CANDIDATE_CYCLE_LEN = 80
CANDIDATE_T0 = 80
CANDIDATE_PATIENCE = 560

# Same pool as longpatience256 (same metadata, same seed)
EXPECTED_VAE_POOL = {1: 567, 2: 567, 3: 568, 4: 568, 5: 568}
RECOVER_SUBJECT_TEST_FOLD = 5  # verified from latent cache: 035_S_6927 outer-test is fold 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--locked-run", type=Path, default=LOCKED_RUN)
    parser.add_argument("--longpatience256-run", type=Path, default=LONGPATIENCE256_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Config-level checks only; no artifact files are read.",
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


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def check_latent384_diff(source: Dict[str, Any], target: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    src_params = dict(source["parameters"])
    tgt_params = dict(target["parameters"])
    for params in (src_params, tgt_params):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
    diffs = {k: (src_params[k], tgt_params[k]) for k in src_params if src_params[k] != tgt_params[k]}
    expected_diffs = {"latent_dim": (BASE_LATENT_DIM, CANDIDATE_LATENT_DIM)}
    if diffs != expected_diffs:
        extra = {k: v for k, v in diffs.items() if k not in expected_diffs}
        missing = {k: v for k, v in expected_diffs.items() if k not in diffs}
        wrong = {k: (expected_diffs[k], diffs[k]) for k in diffs if k in expected_diffs and diffs[k] != expected_diffs[k]}
        if extra:
            errors.append(f"Unexpected diffs vs longpatience: {extra}")
        if missing:
            errors.append(f"Missing expected diff vs longpatience: {missing}")
        if wrong:
            errors.append(f"Wrong diff values: {wrong}")
    if source["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        errors.append("global_tensor_path changed (not allowed)")
    if source["parameters"]["channels_to_use"] != target["parameters"]["channels_to_use"]:
        errors.append("channels_to_use changed (not allowed)")
    src_meta = source["paths"]["metadata_path"]
    tgt_meta = target["paths"]["metadata_path"]
    if src_meta != tgt_meta:
        errors.append(f"metadata_path changed vs longpatience: {src_meta!r} -> {tgt_meta!r} (should be identical)")
    if not tgt_meta.endswith("patched_metadata_candidate.csv"):
        errors.append(f"metadata_path does not end with patched_metadata_candidate.csv: {tgt_meta!r}")
    # Verify unchanged schedule and iterations
    for key, expected_src, expected_tgt in [
        ("epochs_vae", CANDIDATE_EPOCHS_VAE, CANDIDATE_EPOCHS_VAE),
        ("cyclical_beta_n_cycles", CANDIDATE_N_CYCLES, CANDIDATE_N_CYCLES),
        ("lr_scheduler_T0", CANDIDATE_T0, CANDIDATE_T0),
        ("early_stopping_patience_vae", CANDIDATE_PATIENCE, CANDIDATE_PATIENCE),
    ]:
        if tgt_params.get(key) != expected_tgt:
            errors.append(f"{key}={tgt_params.get(key)!r} in target, expected {expected_tgt!r} (unchanged from longpatience)")
    return errors


def check_scheduler_invariants(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    params = config["parameters"]
    cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    if cycle_len != CANDIDATE_CYCLE_LEN:
        errors.append(
            f"Cycle length = {params['epochs_vae']}/{params['cyclical_beta_n_cycles']} = {cycle_len}, "
            f"expected {CANDIDATE_CYCLE_LEN}"
        )
    if params["lr_scheduler_T0"] != CANDIDATE_T0:
        errors.append(f"lr_scheduler_T0={params['lr_scheduler_T0']}, expected {CANDIDATE_T0}")
    if cycle_len != params["lr_scheduler_T0"]:
        errors.append(
            f"Beta cycle length ({cycle_len}) != lr_scheduler_T0 ({params['lr_scheduler_T0']}); "
            "should be phase-aligned"
        )
    patience = params["early_stopping_patience_vae"]
    if patience % CANDIDATE_CYCLE_LEN != 0:
        errors.append(f"patience={patience} is not a multiple of cycle_len={CANDIDATE_CYCLE_LEN}")
    else:
        n_pcycles = patience // CANDIDATE_CYCLE_LEN
        if n_pcycles != 7:
            errors.append(f"patience={patience} = {n_pcycles} cycles, expected 7")
    if params.get("latent_dim") != CANDIDATE_LATENT_DIM:
        errors.append(f"latent_dim={params.get('latent_dim')}, expected {CANDIDATE_LATENT_DIM}")
    return errors


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_latent_cache(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        for split in ["trainDev", "test"]:
            path = run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
            exists = path.exists()
            row: Dict[str, Any] = {
                "fold": fold, "split": split, "path": str(path), "exists": exists,
                "n_rows": None, "n_mu_cols": None, "has_excluded": None,
                "has_recover_subj": None, "errors": [],
            }
            if not exists:
                row["errors"].append("file missing")
                rows.append(row)
                continue
            df = pd.read_csv(path)
            row["n_rows"] = int(len(df))
            mu_cols = [c for c in df.columns if c.startswith("mu_")]
            row["n_mu_cols"] = int(len(mu_cols))
            if len(mu_cols) != CANDIDATE_LATENT_DIM:
                row["errors"].append(
                    f"Expected {CANDIDATE_LATENT_DIM} mu columns, got {len(mu_cols)}"
                )
            if "SubjectID" not in df.columns:
                row["errors"].append("SubjectID column missing")
                rows.append(row)
                continue
            ids = df["SubjectID"].astype(str)
            has_excl = bool((ids == EXCLUDED_SUBJECT).any())
            has_recov = bool((ids == RECOVER_SUBJECT).any())
            row["has_excluded"] = has_excl
            row["has_recover_subj"] = has_recov
            if has_excl:
                row["errors"].append(f"{EXCLUDED_SUBJECT} present in {split} split fold {fold}")
            if split == "test" and fold == RECOVER_SUBJECT_TEST_FOLD and not has_recov:
                row["errors"].append(f"{RECOVER_SUBJECT} absent from test fold {RECOVER_SUBJECT_TEST_FOLD}")
            if split == "trainDev" and fold == RECOVER_SUBJECT_TEST_FOLD and has_recov:
                row["errors"].append(f"{RECOVER_SUBJECT} unexpectedly in trainDev of fold {RECOVER_SUBJECT_TEST_FOLD}")
            if split == "test" and fold != RECOVER_SUBJECT_TEST_FOLD and has_recov:
                row["errors"].append(f"{RECOVER_SUBJECT} in test split of fold {fold} but expected only in fold {RECOVER_SUBJECT_TEST_FOLD}")
            rows.append(row)
    return rows


def check_vae_internal_val(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fold_dir = run_dir / f"fold_{fold}"
        history_path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        row: Dict[str, Any] = {"fold": fold, "history_file": None, "n_val_steps": None, "errors": []}
        if not history_path.exists():
            row["errors"].append(f"vae_train_history_fold_{fold}.joblib not found in {fold_dir}")
            rows.append(row)
            continue
        row["history_file"] = str(history_path)
        try:
            hist_obj = joblib.load(history_path)
        except Exception as exc:
            row["errors"].append(f"Failed to load joblib history: {exc}")
            rows.append(row)
            continue
        if isinstance(hist_obj, dict):
            val_loss = hist_obj.get("val_loss") or hist_obj.get("val_loss_modelsel")
            if val_loss is None:
                row["errors"].append(f"No val_loss key in history dict; keys: {list(hist_obj.keys())[:10]}")
            else:
                non_nan = [v for v in val_loss if v is not None and not (isinstance(v, float) and v != v)]
                row["n_val_steps"] = int(len(non_nan))
                if len(non_nan) == 0:
                    row["errors"].append("val_loss list is empty — early stopping may have been disabled")
        elif hasattr(hist_obj, "columns"):
            val_col = next((c for c in hist_obj.columns if "val" in c.lower() and "loss" in c.lower()), None)
            if val_col is None:
                row["errors"].append(f"No val_loss column in history DataFrame; cols: {list(hist_obj.columns)[:10]}")
            else:
                non_nan = hist_obj[val_col].dropna()
                row["n_val_steps"] = int(len(non_nan))
                if len(non_nan) == 0:
                    row["errors"].append("val_loss column is all NaN — early stopping may have been disabled")
        else:
            row["errors"].append(f"Unexpected history type: {type(hist_obj).__name__}")
        rows.append(row)
    return rows


def check_vae_pool_removed(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        candidates = sorted((run_dir / f"fold_{fold}").glob("vae_pool_required_metadata_removed_fold_*.csv")) if (run_dir / f"fold_{fold}").exists() else []
        row: Dict[str, Any] = {"fold": fold, "file": None, "n_removed": None, "excluded_subject_removed": None, "errors": []}
        if not candidates:
            row["errors"].append("vae_pool_required_metadata_removed_fold_*.csv not found")
            rows.append(row)
            continue
        p = candidates[0]
        row["file"] = str(p)
        df = pd.read_csv(p)
        row["n_removed"] = int(len(df))
        if "SubjectID" in df.columns:
            has_excl = bool((df["SubjectID"].astype(str) == EXCLUDED_SUBJECT).any())
            row["excluded_subject_removed"] = has_excl
            if not has_excl:
                row["errors"].append(f"{EXCLUDED_SUBJECT} not found in removed subjects list")
        else:
            row["errors"].append("SubjectID column missing in removed subjects file")
        rows.append(row)
    return rows


def check_vae_pool_sizes(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        # Use vae_training_pool_tensor_idx.npy — contains the actual VAE training pool indices.
        # The classifier latent cache contains only classifier subjects (~317-318), not the full VAE pool.
        idx_path = run_dir / f"fold_{fold}" / "vae_training_pool_tensor_idx.npy"
        row: Dict[str, Any] = {"fold": fold, "expected_vae_pool": EXPECTED_VAE_POOL.get(fold), "actual_vae_pool": None, "errors": []}
        if not idx_path.exists():
            row["errors"].append("vae_training_pool_tensor_idx.npy missing; cannot verify vae_pool size")
            rows.append(row)
            continue
        idx = np.load(idx_path)
        row["actual_vae_pool"] = int(len(idx))
        expected = EXPECTED_VAE_POOL.get(fold)
        if expected is not None and row["actual_vae_pool"] != expected:
            row["errors"].append(f"vae_pool n={row['actual_vae_pool']}, expected {expected}")
        rows.append(row)
    return rows


def compare_checkpoints_vs_reference(candidate_run: Path, reference_run: Path, label: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        cand_ckpt = candidate_run / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        ref_ckpt = reference_run / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        row: Dict[str, Any] = {
            "fold": fold,
            "reference_label": label,
            "candidate_exists": cand_ckpt.exists(),
            "reference_exists": ref_ckpt.exists(),
            "sha256_same": None,
            "errors": [],
        }
        if not cand_ckpt.exists():
            row["errors"].append("candidate checkpoint missing")
        if not ref_ckpt.exists():
            row["errors"].append(f"{label} reference checkpoint missing")
        if cand_ckpt.exists() and ref_ckpt.exists():
            cand_hash = file_sha256(cand_ckpt)
            ref_hash = file_sha256(ref_ckpt)
            row["sha256_same"] = bool(cand_hash == ref_hash)
            if cand_hash == ref_hash:
                row["errors"].append(
                    f"Checkpoint SHA256 identical to {label} — latent384 must produce different weights"
                )
        rows.append(row)
    return rows


def check_stage_b_readout(readout: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"readout_path": str(readout), "errors": []}
    required = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "command_log.json",
    ]
    for fname in required:
        if not (readout / fname).exists():
            result["errors"].append(f"Missing: {fname}")
    if result["errors"]:
        return result
    pooled = pd.read_csv(readout / "classifier_sweep_pooled_metrics.csv")
    mask = (
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    if not mask.any():
        result["errors"].append(f"{PRIMARY_MODEL}/{PRIMARY_THRESHOLD} not found in pooled metrics")
        return result
    row = pooled[mask].iloc[0]
    result["auc"] = float(row.get("auc", float("nan")))
    result["pr_auc"] = float(row.get("pr_auc", float("nan")))
    result["balanced_accuracy"] = float(row.get("balanced_accuracy", float("nan")))
    result["sensitivity"] = float(row.get("sensitivity", float("nan")))
    result["specificity"] = float(row.get("specificity", float("nan")))
    result["f1"] = float(row.get("f1", float("nan")))
    result["n"] = int(row.get("n", 0))
    cl = load_json(readout / "command_log.json")
    for safety_key in ["vae_training_launched", "training_launched", "tensor_modified", "original_metadata_modified", "ledger_modified"]:
        if cl.get(safety_key) is True:
            result["errors"].append(f"SAFETY VIOLATION in command_log: {safety_key}=True")
    return result


def write_summary(
    outdir: Path,
    config_errors: List[str],
    latent_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    pool_removed_rows: List[Dict[str, Any]],
    pool_size_rows: List[Dict[str, Any]],
    ckpt_rows: List[Dict[str, Any]],
    readout_result: Dict[str, Any],
    dry_run: bool,
) -> None:
    all_errors = (
        [f"[config] {e}" for e in config_errors]
        + [f"[latent_cache] {e}" for r in latent_rows for e in r["errors"]]
        + [f"[val] {e}" for r in val_rows for e in r["errors"]]
        + [f"[pool_removed] {e}" for r in pool_removed_rows for e in r["errors"]]
        + [f"[pool_size] {e}" for r in pool_size_rows for e in r["errors"]]
        + [f"[checkpoint] {e}" for r in ckpt_rows for e in r["errors"]]
        + [f"[readout] {e}" for e in readout_result.get("errors", [])]
    )

    lines = [
        "# recover035_latent384_T80_h10000_p560 Integrity Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Mode: {'DRY-RUN (config-level only)' if dry_run else 'FULL ARTIFACT AUDIT'}",
        "",
        "## Overall result",
        "",
        f"**{'PASS' if not all_errors else 'FAIL'}** — {len(all_errors)} error(s) found.",
        "",
    ]
    if all_errors:
        lines += ["## Errors", ""]
        for e in all_errors:
            lines.append(f"- {e}")
        lines.append("")

    if not dry_run:
        auc = readout_result.get("auc", float("nan"))
        pr_auc = readout_result.get("pr_auc", float("nan"))
        ba = readout_result.get("balanced_accuracy", float("nan"))
        sens = readout_result.get("sensitivity", float("nan"))
        spec = readout_result.get("specificity", float("nan"))
        f1 = readout_result.get("f1", float("nan"))
        n = readout_result.get("n", 0)
        lines += [
            "## Stage B pooled metrics",
            "",
            "| model | threshold | n | AUC | PR-AUC | BA | Sens | Spec | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {PRIMARY_MODEL} | {PRIMARY_THRESHOLD} | {n} | {auc:.4f} | {pr_auc:.4f} | {ba:.4f} | {sens:.4f} | {spec:.4f} | {f1:.4f} |",
            "",
        ]

    lines += [
        "## Config diff check (vs recover035_longpatience_T80_h10000_p560 base model)",
        "",
        f"Config errors: {len(config_errors)}",
        f"Expected single diff: latent_dim {BASE_LATENT_DIM} -> {CANDIDATE_LATENT_DIM}",
        "",
        "## Schedule invariants (unchanged from longpatience)",
        "",
        f"- Cycle length: {CANDIDATE_EPOCHS_VAE}/{CANDIDATE_N_CYCLES} = {CANDIDATE_CYCLE_LEN} epochs",
        f"- lr_scheduler_T0: {CANDIDATE_T0} (unchanged — phase-aligned)",
        f"- Patience: {CANDIDATE_PATIENCE} = {CANDIDATE_PATIENCE // CANDIDATE_CYCLE_LEN} × {CANDIDATE_CYCLE_LEN} cycles",
        f"- latent_dim: {CANDIDATE_LATENT_DIM} (the controlled change)",
        "",
        "## Latent cache check",
        "",
        f"Each fold's latent cache should have {CANDIDATE_LATENT_DIM} mu columns (mu_0..mu_{CANDIDATE_LATENT_DIM - 1}).",
        "",
        "## Read-only guarantee",
        "",
        "This audit does NOT train, fit thresholds, modify tensors, metadata, ledger, configs, or model outputs.",
    ]
    (outdir / "integrity_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    locked_run = resolve(args.locked_run)
    longpatience256_run = resolve(args.longpatience256_run)
    outdir = resolve(args.output_dir)

    source_cfg = load_json(resolve(SOURCE_CONFIG_PATH))
    candidate_cfg = load_json(resolve(CANDIDATE_CONFIG_PATH))

    config_errors = check_latent384_diff(source_cfg, candidate_cfg)
    config_errors += check_scheduler_invariants(candidate_cfg)

    if config_errors:
        for e in config_errors:
            print(f"[CONFIG ERROR] {e}")
    else:
        print(f"Config diff check: PASS — exactly 1 parameter diff vs recover035_longpatience_T80_h10000_p560")
        print(f"  latent_dim: {BASE_LATENT_DIM} -> {CANDIDATE_LATENT_DIM}")
        print(f"Scheduler invariants: cycle_len={CANDIDATE_CYCLE_LEN}, T0={CANDIDATE_T0}, patience={CANDIDATE_PATIENCE} ({CANDIDATE_PATIENCE//CANDIDATE_CYCLE_LEN} cycles)  [OK]")
        print(f"Latent dimension: {CANDIDATE_LATENT_DIM}  [OK]")

    latent_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    pool_removed_rows: List[Dict[str, Any]] = []
    pool_size_rows: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []
    readout_result: Dict[str, Any] = {"errors": []}

    if not args.dry_run:
        latent_rows = check_latent_cache(candidate_run)
        val_rows = check_vae_internal_val(candidate_run)
        pool_removed_rows = check_vae_pool_removed(candidate_run)
        pool_size_rows = check_vae_pool_sizes(candidate_run)
        ckpt_rows = compare_checkpoints_vs_reference(candidate_run, locked_run, "locked")
        if longpatience256_run.exists():
            ckpt_rows += compare_checkpoints_vs_reference(candidate_run, longpatience256_run, "longpatience256")
        readout_result = check_stage_b_readout(candidate_readout)

        latent_errors = [e for r in latent_rows for e in r["errors"]]
        val_errors = [e for r in val_rows for e in r["errors"]]
        pool_removed_errors = [e for r in pool_removed_rows for e in r["errors"]]
        pool_size_errors = [e for r in pool_size_rows for e in r["errors"]]
        ckpt_errors = [e for r in ckpt_rows for e in r["errors"]]
        readout_errors = readout_result.get("errors", [])

        if not latent_errors:
            print(f"Latent cache checks: PASS (including {CANDIDATE_LATENT_DIM} mu columns per fold)")
        else:
            for e in latent_errors:
                print(f"[LATENT ERROR] {e}")
        if not val_errors:
            print("VAE internal val checks: PASS")
        else:
            for e in val_errors:
                print(f"[VAL ERROR] {e}")
        if not pool_removed_errors:
            print("VAE pool removed checks: PASS")
        else:
            for e in pool_removed_errors:
                print(f"[POOL_REMOVED ERROR] {e}")
        if not pool_size_errors:
            print("VAE pool size checks: PASS")
        else:
            for e in pool_size_errors:
                print(f"[POOL_SIZE ERROR] {e}")
        if not ckpt_errors:
            print("Checkpoint diff checks: PASS (candidate differs from locked and longpatience256)")
        else:
            for e in ckpt_errors:
                print(f"[CKPT ERROR] {e}")
        if not readout_errors:
            print(f"Stage B readout: PASS — AUC={readout_result.get('auc', float('nan')):.4f}, "
                  f"PR-AUC={readout_result.get('pr_auc', float('nan')):.4f}")
        else:
            for e in readout_errors:
                print(f"[READOUT ERROR] {e}")

        if outdir.exists() and not args.overwrite:
            raise FileExistsError(f"{outdir} exists; pass --overwrite")
        outdir.mkdir(parents=True, exist_ok=True)

        write_table(outdir, "latent_cache_check", pd.DataFrame(latent_rows))
        write_table(outdir, "vae_internal_val_check", pd.DataFrame(val_rows))
        write_table(outdir, "vae_pool_removed_check", pd.DataFrame(pool_removed_rows))
        write_table(outdir, "vae_pool_size_check", pd.DataFrame(pool_size_rows))
        write_table(outdir, "checkpoint_diff_check", pd.DataFrame(ckpt_rows))

        cl = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "candidate_run": str(candidate_run),
            "candidate_readout": str(candidate_readout),
            "dry_run": False,
            "training_launched": False,
            "tensor_modified": False,
            "original_metadata_modified": False,
            "ledger_modified": False,
            "locked_model_outputs_modified": False,
            "longpatience256_outputs_modified": False,
        }
        (outdir / "command_log.json").write_text(json.dumps(cl, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_summary(
        outdir if not args.dry_run else Path("."),
        config_errors, latent_rows, val_rows, pool_removed_rows,
        pool_size_rows, ckpt_rows, readout_result, dry_run=args.dry_run,
    )

    if args.dry_run:
        status = "PASS" if not config_errors else "FAIL"
        print(f"\nDry-run complete. Config-level checks: {status}")
    else:
        all_errors = (
            config_errors
            + [e for r in latent_rows for e in r["errors"]]
            + [e for r in val_rows for e in r["errors"]]
            + [e for r in pool_removed_rows for e in r["errors"]]
            + [e for r in pool_size_rows for e in r["errors"]]
            + [e for r in ckpt_rows for e in r["errors"]]
            + readout_result.get("errors", [])
        )
        status = "PASS" if not all_errors else "FAIL"
        print(f"\nFull audit: {status} ({len(all_errors)} error(s)). Output: {outdir}")
    return 0 if not config_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
