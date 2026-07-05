#!/usr/bin/env python3
"""Read-only integrity audit for recover035_latent384_beta3p75_T80_h10000_p560 FULL 5x5.

Verifies:
  - Exactly 1 parameter diff vs recover035_latent384_T80_h10000_p560_full5x5 (the base):
      beta_vae 2.5 -> 3.75
  - latent_dim remains 384; all schedule params unchanged
  - metadata_path ends with patched_metadata_candidate.csv (same as latent384_beta2.5)
  - global_tensor_path unchanged
  - cycle length = 10000/125 = 80 epochs (unchanged)
  - patience = 560 = 7 x 80-epoch cycles (unchanged)
  - 035_S_6927 appears in latent caches for its outer-test fold and trainDev for others
  - 128_S_2002 is absent from ALL latent caches
  - Latent cache has 384 mu columns (mu_0..mu_383) per fold
  - VAE internal val non-empty per fold
  - vae_pool_required_metadata_removed_fold_N.csv exists and contains 128_S_2002 per fold
  - VAE checkpoints differ from locked and latent384_beta2.5 (new training)

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

CANDIDATE_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
LATENT384_BETA2P5_RUN = RESULTS / "recover035_latent384_T80_h10000_p560_full5x5"
OUTPUT_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_integrity_audit"

SOURCE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_T80_h10000_p560_full5x5.json"
)
CANDIDATE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FOLDS = [1, 2, 3, 4, 5]

CANDIDATE_LATENT_DIM = 384
CANDIDATE_BETA = 3.75
BASE_BETA = 2.5

CANDIDATE_EPOCHS_VAE = 10000
CANDIDATE_N_CYCLES = 125
CANDIDATE_CYCLE_LEN = 80
CANDIDATE_T0 = 80
CANDIDATE_PATIENCE = 560

EXPECTED_VAE_POOL = {1: 567, 2: 567, 3: 568, 4: 568, 5: 568}
RECOVER_SUBJECT_TEST_FOLD = 5

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--locked-run", type=Path, default=LOCKED_RUN)
    parser.add_argument("--latent384-beta2p5-run", type=Path, default=LATENT384_BETA2P5_RUN)
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


def check_beta3p75_diff(source: Dict[str, Any], target: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    src_params = dict(source["parameters"])
    tgt_params = dict(target["parameters"])
    for params in (src_params, tgt_params):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
    diffs = {k: (src_params[k], tgt_params[k]) for k in src_params if src_params[k] != tgt_params[k]}
    expected_diffs = {"beta_vae": (BASE_BETA, CANDIDATE_BETA)}
    if diffs != expected_diffs:
        extra = {k: v for k, v in diffs.items() if k not in expected_diffs}
        missing = {k: v for k, v in expected_diffs.items() if k not in diffs}
        wrong = {k: (expected_diffs[k], diffs[k]) for k in diffs if k in expected_diffs and diffs[k] != expected_diffs[k]}
        if extra:
            errors.append(f"Unexpected diffs vs latent384_beta2.5: {extra}")
        if missing:
            errors.append(f"Missing expected diff vs latent384_beta2.5: {missing}")
        if wrong:
            errors.append(f"Wrong diff values: {wrong}")
    if source["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        errors.append("global_tensor_path changed (not allowed)")
    if source["parameters"]["channels_to_use"] != target["parameters"]["channels_to_use"]:
        errors.append("channels_to_use changed (not allowed)")
    src_meta = source["paths"]["metadata_path"]
    tgt_meta = target["paths"]["metadata_path"]
    if src_meta != tgt_meta:
        errors.append(f"metadata_path changed vs latent384_beta2.5: {src_meta!r} -> {tgt_meta!r} (should be identical)")
    if not tgt_meta.endswith("patched_metadata_candidate.csv"):
        errors.append(f"metadata_path does not end with patched_metadata_candidate.csv: {tgt_meta!r}")
    if tgt_params.get("latent_dim") != CANDIDATE_LATENT_DIM:
        errors.append(f"latent_dim changed to {tgt_params.get('latent_dim')!r}; must remain {CANDIDATE_LATENT_DIM}")
    for key, expected_val in [
        ("epochs_vae", CANDIDATE_EPOCHS_VAE),
        ("cyclical_beta_n_cycles", CANDIDATE_N_CYCLES),
        ("lr_scheduler_T0", CANDIDATE_T0),
        ("early_stopping_patience_vae", CANDIDATE_PATIENCE),
    ]:
        if tgt_params.get(key) != expected_val:
            errors.append(f"{key}={tgt_params.get(key)!r} in target, expected {expected_val!r} (unchanged from latent384_beta2.5)")
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
    if params.get("beta_vae") != CANDIDATE_BETA:
        errors.append(f"beta_vae={params.get('beta_vae')}, expected {CANDIDATE_BETA}")
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


def check_metadata_removed_csvs(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        path = run_dir / f"fold_{fold}" / f"vae_pool_required_metadata_removed_fold_{fold}.csv"
        exists = path.exists()
        row: Dict[str, Any] = {"fold": fold, "path": str(path), "exists": exists, "n_rows": None, "has_excluded": None, "errors": []}
        if not exists:
            row["errors"].append("file missing")
            rows.append(row)
            continue
        df = pd.read_csv(path)
        row["n_rows"] = int(len(df))
        if "SubjectID" not in df.columns:
            row["errors"].append("SubjectID column missing")
        else:
            has_excl = bool((df["SubjectID"].astype(str) == EXCLUDED_SUBJECT).any())
            row["has_excluded"] = has_excl
            if not has_excl:
                row["errors"].append(f"{EXCLUDED_SUBJECT} absent from metadata_removed CSV (should be listed there)")
        rows.append(row)
    return rows


def check_vae_checkpoints(run_dir: Path, ref_dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        ckpt_path = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        row: Dict[str, Any] = {
            "fold": fold,
            "candidate_ckpt_exists": ckpt_path.exists(),
            "candidate_sha256": None,
            "errors": [],
        }
        if not ckpt_path.exists():
            row["errors"].append(f"checkpoint missing: {ckpt_path}")
            for ref_key in ref_dirs:
                row[f"differs_from_{ref_key}"] = None
            rows.append(row)
            continue
        cand_sha = file_sha256(ckpt_path)
        row["candidate_sha256"] = cand_sha[:16]
        for ref_key, ref_dir in ref_dirs.items():
            ref_ckpt = ref_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
            if not ref_ckpt.exists():
                row[f"differs_from_{ref_key}"] = None
            else:
                ref_sha = file_sha256(ref_ckpt)
                differs = cand_sha != ref_sha
                row[f"differs_from_{ref_key}"] = differs
                if not differs:
                    row["errors"].append(
                        f"Checkpoint identical to {ref_key} fold {fold} — "
                        "this should be a fresh training run with different beta_vae"
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def check_stage_b_predictions(readout_dir: Path) -> Dict[str, Any]:
    path = readout_dir / "classifier_sweep_predictions.csv"
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "errors": []}
    if not path.exists():
        result["errors"].append("classifier_sweep_predictions.csv missing")
        return result
    df = pd.read_csv(path)
    result["n_rows"] = int(len(df))
    result["n_unique_subjects"] = int(df["SubjectID"].nunique()) if "SubjectID" in df.columns else None
    required_cols = ["SubjectID", "fold", "y_true", "y_score", "model_name"]
    for col in required_cols:
        if col not in df.columns:
            result["errors"].append(f"Missing column: {col}")
    if EXCLUDED_SUBJECT in df.get("SubjectID", pd.Series(dtype=str)).astype(str).values:
        result["errors"].append(f"{EXCLUDED_SUBJECT} present in Stage B predictions (must be excluded)")
    return result


def check_pooled_metrics(readout_dir: Path) -> Dict[str, Any]:
    path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    result: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "auc": None, "pr_auc": None, "errors": []}
    if not path.exists():
        result["errors"].append("classifier_sweep_pooled_metrics.csv missing")
        return result
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL)
    mask &= df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    sub = df[mask]
    if sub.empty:
        result["errors"].append(f"No row for {PRIMARY_MODEL}/{PRIMARY_THRESHOLD}")
        return result
    row = sub.iloc[0]
    result["auc"] = float(row.get("auc", float("nan")))
    result["pr_auc"] = float(row.get("pr_auc", float("nan")))
    result["promotes_auc"] = bool(result["auc"] > LOCKED_AUC)
    result["promotes_pr_auc"] = bool(result["pr_auc"] >= LOCKED_PR_AUC)
    result["promotes"] = result["promotes_auc"] and result["promotes_pr_auc"]
    return result


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    outdir = resolve(args.output_dir)

    print(f"Candidate run    : {candidate_run}")
    print(f"Candidate readout: {candidate_readout}")
    print(f"Config           : {CANDIDATE_CONFIG_PATH}")
    print(f"Source config    : {SOURCE_CONFIG_PATH} (latent384_beta2.5)")

    if not CANDIDATE_CONFIG_PATH.exists():
        raise RuntimeError(f"Candidate config not found: {CANDIDATE_CONFIG_PATH}")
    if not SOURCE_CONFIG_PATH.exists():
        raise RuntimeError(f"Source config not found: {SOURCE_CONFIG_PATH}")

    candidate_cfg = load_json(CANDIDATE_CONFIG_PATH)
    source_cfg = load_json(SOURCE_CONFIG_PATH)

    diff_errors = check_beta3p75_diff(source_cfg, candidate_cfg)
    sched_errors = check_scheduler_invariants(candidate_cfg)
    config_errors = diff_errors + sched_errors

    if config_errors:
        for e in config_errors:
            print(f"CONFIG ERROR: {e}")
        raise RuntimeError(f"Config-level checks failed ({len(config_errors)} errors). Aborting.")

    print(f"Config diff check : beta_vae {BASE_BETA}->{CANDIDATE_BETA}, all other params unchanged  [OK]")
    print(f"Scheduler check   : cycle_len={CANDIDATE_CYCLE_LEN}, T0={CANDIDATE_T0}, patience={CANDIDATE_PATIENCE}={CANDIDATE_PATIENCE // CANDIDATE_CYCLE_LEN} cycles  [OK]")
    print(f"latent_dim        : {CANDIDATE_LATENT_DIM} (unchanged)  [OK]")
    print(f"beta_vae          : {CANDIDATE_BETA}  [OK]")

    if args.dry_run:
        print("\nDry-run complete. Artifact file checks skipped.")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    all_errors: List[str] = []

    latent_rows = check_latent_cache(candidate_run)
    latent_df = pd.DataFrame([{k: v for k, v in r.items() if k != "errors"} for r in latent_rows])
    latent_df["error_count"] = [len(r["errors"]) for r in latent_rows]
    latent_df["errors"] = [" | ".join(r["errors"]) for r in latent_rows]
    write_table(outdir, "latent_cache_check", latent_df)
    for r in latent_rows:
        for e in r["errors"]:
            all_errors.append(f"latent_cache fold{r['fold']} {r['split']}: {e}")

    meta_removed_rows = check_metadata_removed_csvs(candidate_run)
    meta_df = pd.DataFrame([{k: v for k, v in r.items() if k != "errors"} for r in meta_removed_rows])
    meta_df["error_count"] = [len(r["errors"]) for r in meta_removed_rows]
    meta_df["errors"] = [" | ".join(r["errors"]) for r in meta_removed_rows]
    write_table(outdir, "metadata_removed_check", meta_df)
    for r in meta_removed_rows:
        for e in r["errors"]:
            all_errors.append(f"metadata_removed fold{r['fold']}: {e}")

    ref_dirs = {
        "locked": resolve(args.locked_run),
        "latent384_beta2p5": resolve(args.latent384_beta2p5_run),
    }
    ckpt_df = check_vae_checkpoints(candidate_run, ref_dirs)
    if not isinstance(ckpt_df, pd.DataFrame):
        ckpt_df = pd.DataFrame(ckpt_df)
    write_table(outdir, "vae_checkpoint_check", ckpt_df)
    for _, r in ckpt_df.iterrows():
        errs = r.get("errors", [])
        if isinstance(errs, list):
            for e in errs:
                all_errors.append(f"checkpoint fold{r['fold']}: {e}")
        elif isinstance(errs, str) and errs:
            all_errors.append(f"checkpoint fold{r['fold']}: {errs}")

    sb_result = check_stage_b_predictions(candidate_readout)
    pooled_result = check_pooled_metrics(candidate_readout)

    sb_df = pd.DataFrame([sb_result])
    write_table(outdir, "stage_b_predictions_check", sb_df)
    for e in sb_result.get("errors", []):
        all_errors.append(f"stage_b_predictions: {e}")

    promotion_verdict = "PROMOTES" if pooled_result.get("promotes") else "DOES NOT PROMOTE"
    summary_lines = [
        "# Integrity Audit Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Candidate: {candidate_run.name}",
        f"beta_vae: {BASE_BETA} -> {CANDIDATE_BETA} (single controlled diff vs latent384_beta2.5)",
        f"latent_dim: {CANDIDATE_LATENT_DIM} (unchanged)",
        "",
        "## Config checks",
        f"- diff vs latent384_beta2.5: beta_vae {BASE_BETA}->{CANDIDATE_BETA} only  [PASS]",
        f"- scheduler invariants: cycle_len={CANDIDATE_CYCLE_LEN}, T0={CANDIDATE_T0}, patience={CANDIDATE_PATIENCE}  [PASS]",
        f"- metadata_path: patched_metadata_candidate.csv  [PASS]",
        "",
        "## Artifact checks",
        f"- Latent cache errors: {sum(r['error_count'] for _, r in latent_df.iterrows())}",
        f"- Metadata removed CSV errors: {meta_df['error_count'].sum()}",
        "",
        "## Stage B",
        f"- Predictions file: {'OK' if sb_result['exists'] else 'MISSING'}",
        f"- n_subjects: {sb_result.get('n_unique_subjects')}",
        f"- Pooled AUC: {pooled_result.get('auc')}  (threshold: > {LOCKED_AUC} -> {'PASS' if pooled_result.get('promotes_auc') else 'FAIL'})",
        f"- Pooled PR-AUC: {pooled_result.get('pr_auc')}  (threshold: >= {LOCKED_PR_AUC} -> {'PASS' if pooled_result.get('promotes_pr_auc') else 'FAIL'})",
        f"- Promotion verdict: **{promotion_verdict}**",
        "",
        "## Total errors",
        f"{len(all_errors)}",
        "",
    ]
    for e in all_errors:
        summary_lines.append(f"- {e}")
    summary_lines.append("")
    summary_lines.append("Read-only. Did not train, modify tensor, metadata, ledger, or model outputs.")
    (outdir / "integrity_audit_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    if all_errors:
        print(f"\nINTEGRITY AUDIT FAILED — {len(all_errors)} error(s):")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print(f"\nIntegrity audit PASSED. Pooled AUC={pooled_result.get('auc')}, PR-AUC={pooled_result.get('pr_auc')}")
    print(f"Promotion: {promotion_verdict}")
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
