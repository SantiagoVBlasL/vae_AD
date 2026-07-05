#!/usr/bin/env python3
"""Beta sensitivity sweep preflight for ch1-only Pearson model, Site31 Mayo reprocessed14 tensor.

Reference run:
  recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14

Candidates (single controlled diff — only beta_vae changes):
  beta4p5: recover035_ch1only_latent384_beta4p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14
  beta5p5: recover035_ch1only_latent384_beta5p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14

Everything else is locked:
  channels_to_use=[1]  (Pearson_Full_FisherZ_Signed only)
  global_tensor_path=site31_mayo_reprocessed14 (UNCHANGED from reference)
  latent_dim=384, T0=80, epochs=10000, patience=560, batch=64, dropout=0.15
  seed=42, folds=5x5, norm_mode=zscore_offdiag, recon_loss=mse_sum_batchmean_current

Default behavior is preflight/dry-run only.
Real training requires --confirm-training for each candidate individually.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

# ── Reference (ch1only beta3p75 reprocessed14 — the completed run) ──────────
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14.json"
)
REFERENCE_RUN_ID = (
    "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14"
)
REFERENCE_BETA = 3.75

# ── Candidates ───────────────────────────────────────────────────────────────
CANDIDATES: List[Dict[str, Any]] = [
    {
        "beta": 4.5,
        "beta_str": "4p5",
        "config": PROJECT_ROOT
        / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta4p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14.json",
        "run_id": "recover035_ch1only_latent384_beta4p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14",
    },
    {
        "beta": 5.5,
        "beta_str": "5p5",
        "config": PROJECT_ROOT
        / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta5p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14.json",
        "run_id": "recover035_ch1only_latent384_beta5p5_T80_h10000_p560_full5x5_site31_mayo_reprocessed14",
    },
]

DEFAULT_PREFLIGHT_DIR = RESULTS / "ch1only_beta_sweep_reprocessed14_preflight_20260621"

# ── Pipeline scripts ─────────────────────────────────────────────────────────
STAGE_B_SCRIPT = (
    PROJECT_ROOT
    / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
)
OOF_SCORE_SCRIPT = (
    PROJECT_ROOT
    / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"
)

# ── Locked constants ─────────────────────────────────────────────────────────
EXPECTED_TENSOR_PATH = (
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14.npz"
)
ORIGINAL_TENSOR_SHA256 = "f9a00b291a88d92d942ee3404fe0f5b1cfabe5178cf8ff6c139d57658cb8f609"
REPLACEMENT_TENSOR_SHA256 = "9d5cb75f30ab6bae368c738c2ce1fb5175b7630db2b4492067d46c3956267301"
EXPECTED_REPROCESSED_N = 14
EXPECTED_CHANNELS = [1]
EXPECTED_SELECTED_NAMES = ["Pearson_Full_FisherZ_Signed"]

EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
PRIMARY_MODEL = "logreg_l2"

# Parameters that must be IDENTICAL across reference and all candidates.
LOCKED_PARAMS = [
    "channels_to_use",
    "latent_dim",
    "dropout_rate_vae",
    "vae_dropout_scope",
    "vae_block_order",
    "epochs_vae",
    "cyclical_beta_n_cycles",
    "cyclical_beta_ratio_increase",
    "lr_scheduler_T0",
    "early_stopping_patience_vae",
    "batch_size",
    "decoder_type",
    "recon_loss_mode",
    "norm_mode",
    "vae_final_activation",
    "intermediate_fc_dim_vae",
    "classifier_types",
    "classifier_stratify_cols",
    "vae_stratify_cols",
    "metadata_features",
    "seed",
    "n_iter_logreg",
    "n_iter_svm",
    "vae_train_sampler_strategy",
]

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")

PROMOTION_RULE = (
    "Beta sweep candidate promotes only if ALL hold simultaneously: "
    "(1) ADNI pooled AUC > 0.782951 AND PR-AUC >= 0.559873 (locked promotion thresholds). "
    "(2) OASIS external AUC assessed and non-inferior. "
    "(3) Philips CN FPR <= 0.0808 (promoted 3-channel reference) — ref run had FPR=0.4949, "
    "so the required improvement is substantial. "
    "(4) Robustness must demonstrably improve over beta3p75 ch1only reference; "
    "AUC gain alone is insufficient. "
    "No auto-promotion on AUC alone."
)


# ── Utilities ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    p.add_argument("--python-executable", default="/home/diego/anaconda3/envs/vae_ad/bin/python")
    p.add_argument("--confirm-training", metavar="BETA_STR",
                   help="Launch training for one candidate by beta_str (4p5 or 5p5). "
                        "Preflight must pass first.")
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, max_rows: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    view = df.head(max_rows).copy()
    try:
        md = view.to_markdown(index=False)
    except Exception:
        md = view.to_string(index=False)
    if len(df) > max_rows:
        md += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (out_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def append_arg(cmd: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    if value is None:
        return
    cmd.append(f"--{name}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[tokens.index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


# ── Validation functions ─────────────────────────────────────────────────────

def validate_candidate_diff(
    reference: Dict[str, Any],
    target: Dict[str, Any],
    expected_beta: float,
) -> Dict[str, Any]:
    """Enforce exactly one scientific diff: beta_vae = REFERENCE_BETA -> expected_beta."""
    ref_params = dict(reference["parameters"])
    tgt_params = dict(target["parameters"])

    if set(ref_params) != set(tgt_params):
        raise RuntimeError(
            f"Parameter key set mismatch vs reference: "
            f"missing={sorted(set(ref_params) - set(tgt_params))}, "
            f"extra={sorted(set(tgt_params) - set(ref_params))}"
        )

    # All locked params must be identical.
    for key in LOCKED_PARAMS:
        if ref_params[key] != tgt_params[key]:
            raise RuntimeError(
                f"Locked param '{key}' changed: {ref_params[key]!r} -> {tgt_params[key]!r}. "
                f"Only beta_vae is allowed to differ."
            )

    # channels must be [1] in candidate.
    if tgt_params["channels_to_use"] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"channels_to_use must be {EXPECTED_CHANNELS}; got {tgt_params['channels_to_use']}"
        )

    # beta_vae must be the expected sweep value.
    if float(tgt_params["beta_vae"]) != expected_beta:
        raise RuntimeError(
            f"Expected beta_vae={expected_beta}, got {tgt_params['beta_vae']}"
        )
    if float(ref_params["beta_vae"]) != REFERENCE_BETA:
        raise RuntimeError(
            f"Reference beta_vae must be {REFERENCE_BETA}, got {ref_params['beta_vae']}"
        )

    # Tensor path must be the SAME reprocessed14 tensor (no second diff allowed).
    if target["paths"]["global_tensor_path"] != EXPECTED_TENSOR_PATH:
        raise RuntimeError(
            f"global_tensor_path must be the site31_mayo_reprocessed14 tensor (same as reference).\n"
            f"Expected: {EXPECTED_TENSOR_PATH}\nGot: {target['paths']['global_tensor_path']}"
        )
    if reference["paths"]["global_tensor_path"] != EXPECTED_TENSOR_PATH:
        raise RuntimeError(
            f"Reference global_tensor_path is not the reprocessed14 tensor — "
            f"check that reference config is ch1only reprocessed14, not standard tensor."
        )

    # metadata_path must be unchanged.
    if reference["paths"]["metadata_path"] != target["paths"]["metadata_path"]:
        raise RuntimeError("metadata_path changed — must be identical to reference.")

    # selected_channel_names must be unchanged.
    if target.get("selected_channel_names") != EXPECTED_SELECTED_NAMES:
        raise RuntimeError(f"selected_channel_names must be {EXPECTED_SELECTED_NAMES}.")

    return {
        "single_scientific_diff": "beta_vae",
        "reference_beta": REFERENCE_BETA,
        "candidate_beta": expected_beta,
        "channels_to_use_locked": EXPECTED_CHANNELS,
        "tensor_path_locked": EXPECTED_TENSOR_PATH,
        "n_locked_params": len(LOCKED_PARAMS),
    }


def load_metadata(config: Dict[str, Any]) -> pd.DataFrame:
    meta = pd.read_csv(resolve(config["paths"]["metadata_path"]))
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    for col in ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]:
        if col not in meta.columns:
            raise RuntimeError(f"Metadata missing required column: {col}")
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    if meta[["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]].isna().any().any():
        bad = meta[meta[["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]].isna().any(axis=1)]
        raise RuntimeError(f"Metadata has missing required values:\n{bad.head(10).to_string()}")
    return meta


def validate_subject_pool(meta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dx_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf_counts = clf["ResearchGroup_Mapped"].value_counts().to_dict()
    for dx, expected in EXPECTED_VAE_COUNTS.items():
        obs = int(dx_counts.get(dx, 0))
        rows.append({"check": f"vae_{dx}", "observed": obs, "expected": expected, "pass": obs == expected})
    for dx, expected in EXPECTED_CLF_COUNTS.items():
        obs = int(clf_counts.get(dx, 0))
        rows.append({"check": f"classifier_{dx}", "observed": obs, "expected": expected, "pass": obs == expected})
    rows.extend([
        {"check": "035_S_6927_present", "observed": bool((meta["SubjectID"] == RECOVER_SUBJECT).any()),
         "expected": True, "pass": bool((meta["SubjectID"] == RECOVER_SUBJECT).any())},
        {"check": "128_S_2002_absent", "observed": bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()),
         "expected": False, "pass": not bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any())},
        {"check": "classifier_pool_n", "observed": int(len(clf)), "expected": 397, "pass": int(len(clf)) == 397},
        {"check": "vae_pool_n", "observed": int(len(meta)), "expected": 647, "pass": int(len(meta)) == 647},
    ])
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Subject pool validation failed:\n" + out.to_string(index=False))
    return out


def validate_tensor(meta: pd.DataFrame) -> Dict[str, Any]:
    tensor_path = Path(EXPECTED_TENSOR_PATH)
    if not tensor_path.exists():
        raise RuntimeError(f"Replacement tensor not found: {tensor_path}")
    with np.load(tensor_path, allow_pickle=True) as zf:
        tensor_shape = tuple(int(v) for v in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_ids = [str(x) for x in zf["subject_ids"].astype(str)]
        reprocessed_sids = [str(x) for x in zf["site31_mayo_reprocessed_subject_ids"].astype(str)]
        stored_original_sha256 = str(zf["site31_mayo_original_tensor_sha256"])
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    if selected != EXPECTED_SELECTED_NAMES:
        raise RuntimeError(f"Channel name mismatch: got {selected}, expected {EXPECTED_SELECTED_NAMES}")
    if tensor_shape != (648, 7, 131, 131):
        raise RuntimeError(f"Expected tensor shape (648,7,131,131), got {tensor_shape}")
    if len(reprocessed_sids) != EXPECTED_REPROCESSED_N:
        raise RuntimeError(
            f"Expected {EXPECTED_REPROCESSED_N} reprocessed subjects, got {len(reprocessed_sids)}"
        )
    if stored_original_sha256 != ORIGINAL_TENSOR_SHA256:
        raise RuntimeError(
            f"Stored original SHA256 mismatch.\nStored: {stored_original_sha256}\n"
            f"Expected: {ORIGINAL_TENSOR_SHA256}"
        )
    mismatches = sum(
        1 for _, row in meta.iterrows()
        if str(row["SubjectID"]) != subject_ids[int(row["tensor_idx"])]
    )
    if mismatches > 0:
        raise RuntimeError(f"tensor_idx alignment: {mismatches} rows do not match subject_ids in tensor")
    return {
        "tensor_path": str(tensor_path),
        "tensor_shape_full": tensor_shape,
        "selected_channels_to_use": EXPECTED_CHANNELS,
        "selected_channel_names": selected,
        "n_metadata_rows": int(len(meta)),
        "n_tensor_subject_ids": int(len(subject_ids)),
        "n_site31_mayo_reprocessed_subjects": int(len(reprocessed_sids)),
        "site31_mayo_reprocessed_subjects": " | ".join(reprocessed_sids),
        "stored_original_sha256_matches_promoted": bool(stored_original_sha256 == ORIGINAL_TENSOR_SHA256),
        "replacement_tensor_sha256_expected": REPLACEMENT_TENSOR_SHA256,
        "035_S_6927_in_tensor": bool(RECOVER_SUBJECT in subject_ids),
        "128_S_2002_in_tensor": bool(EXCLUDED_SUBJECT in subject_ids),
        "tensor_idx_alignment_mismatches": mismatches,
        "tensor_modification_status": "UNTOUCHED — sweep uses same read-only replacement tensor",
        "shared_across_candidates": True,
    }


def split_key(df: pd.DataFrame) -> pd.Series:
    return df["ResearchGroup_Mapped"].astype(str) + "_" + df["Manufacturer"].astype(str)


def count_dx_mfr(df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "n": int(len(df)),
        "CN": int((df["ResearchGroup_Mapped"] == "CN").sum()),
        "MCI": int((df["ResearchGroup_Mapped"] == "MCI").sum()),
        "AD": int((df["ResearchGroup_Mapped"] == "AD").sum()),
    }
    for mfr in sorted(df["Manufacturer"].astype(str).unique()):
        row[f"Manufacturer_{mfr}"] = int((df["Manufacturer"].astype(str) == mfr).sum())
    return row


def validate_fold_feasibility(config: Dict[str, Any], meta: pd.DataFrame) -> pd.DataFrame:
    params = config["parameters"]
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    outer_key = split_key(clf)
    min_outer = int(outer_key.value_counts().min())
    if min_outer < int(params["outer_folds"]):
        raise RuntimeError(f"Outer stratification infeasible: min stratum {min_outer}")
    outer = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    rows: List[Dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(clf, outer_key), start=1):
        train_dev = clf.iloc[train_idx].copy()
        test = clf.iloc[test_idx].copy()
        test_subjects = set(test["SubjectID"].astype(str))
        vae_pool = meta[~meta["SubjectID"].astype(str).isin(test_subjects)].copy()
        vae_key = split_key(vae_pool)
        min_vae_stratum = int(vae_key.value_counts().min())
        val_ok = True
        val_error = ""
        try:
            train_test_split(
                np.arange(len(vae_pool)),
                test_size=float(params["vae_val_split_ratio"]),
                random_state=int(params["seed"]) + fold,
                stratify=vae_key if min_vae_stratum >= 2 else None,
            )
        except Exception as exc:
            val_ok = False
            val_error = str(exc)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test), ("vae_pool", vae_pool)]:
            row = {"fold": fold, "split_component": split_name}
            row.update(count_dx_mfr(df))
            row["all_dx_required_present"] = (
                bool(row["CN"] > 0 and row["AD"] > 0)
                if split_name.startswith("classifier")
                else bool(row["CN"] > 0 and row["MCI"] > 0 and row["AD"] > 0)
            )
            row["all_three_mfr_present"] = all(
                row.get(f"Manufacturer_{m}", 0) > 0 for m in ["GE", "Philips", "SIEMENS"]
            )
            row["outer_test_overlap_in_vae_pool"] = int(len(test_subjects & set(vae_pool["SubjectID"].astype(str))))
            row["vae_val_split_feasible"] = val_ok if split_name == "vae_pool" else ""
            row["vae_val_split_error"] = val_error if split_name == "vae_pool" else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    failed = out[
        (~out["all_dx_required_present"])
        | (~out["all_three_mfr_present"])
        | (out["outer_test_overlap_in_vae_pool"] != 0)
        | ((out["split_component"] == "vae_pool") & (out["vae_val_split_feasible"] != True))
    ]
    if not failed.empty:
        raise RuntimeError("Fold feasibility failed:\n" + failed.to_string(index=False))
    return out


def output_symlink_status(config: Dict[str, Any]) -> Dict[str, Any]:
    local = resolve(config["paths"]["output_dir"])
    target = Path(config["paths"]["big_disk_output_dir"])
    local_exists = local.exists() or local.is_symlink()
    stale: List[Path] = []
    if local_exists and (local.exists() and local.is_dir()):
        for child in local.iterdir():
            if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
                stale.append(child)
    return {
        "local_output_dir": str(local),
        "big_disk_output_dir": str(target),
        "big_disk_parent_exists": bool(target.parent.exists()),
        "big_disk_target_exists": bool(target.exists()),
        "local_exists": bool(local_exists),
        "local_is_symlink": bool(local.is_symlink()),
        "local_is_regular_directory": bool(local.exists() and local.is_dir() and not local.is_symlink()),
        "symlink_target": str(local.resolve()) if local_exists else "",
        "target_match": bool(local_exists and local.is_symlink() and local.resolve() == target.resolve()),
        "preflight_clean_missing_output_ok": bool((not local_exists) and (not target.exists()) and target.parent.exists()),
        "stale_marker_count": len(stale),
        "stale_markers": " | ".join(str(p) for p in sorted(stale)),
    }


# ── Command builders ─────────────────────────────────────────────────────────

def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    cmd = [
        python_exe,
        str(resolve(config["paths"]["training_script"])),
        "--global_tensor_path", str(resolve(config["paths"]["global_tensor_path"])),
        "--metadata_path", str(resolve(config["paths"]["metadata_path"])),
        "--output_dir", str(resolve(config["paths"]["output_dir"])),
    ]
    for key, value in config["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    return cmd


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe, str(STAGE_B_SCRIPT),
        "--run-dir", str(out),
        "--output-dir", str(out / "classifier_only_readout"),
        "--outer-folds", str(config["parameters"]["outer_folds"]),
        "--inner-folds", str(config["parameters"]["inner_folds"]),
        "--models", PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]


def build_oof_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    run_out = resolve(config["paths"]["output_dir"])
    oof_out = RESULTS / f"{config['run_name']}_stageB_oof_score_calibration"
    return [
        python_exe, str(OOF_SCORE_SCRIPT),
        "--run-dir", str(run_out),
        "--output-dir", str(oof_out),
    ]


def validate_commands(
    config: Dict[str, Any],
    stage_a: Sequence[str],
    stage_b: Sequence[str],
    oof: Sequence[str],
    expected_beta: float,
) -> pd.DataFrame:
    beta_str = str(expected_beta)
    checks = [
        ("stage_a_channels", values_after_flag(stage_a, "--channels_to_use"), ["1"]),
        ("stage_a_latent_dim", values_after_flag(stage_a, "--latent_dim"), ["384"]),
        ("stage_a_beta", values_after_flag(stage_a, "--beta_vae"), [beta_str]),
        ("stage_a_recon_loss", values_after_flag(stage_a, "--recon_loss_mode"), ["mse_sum_batchmean_current"]),
        ("stage_a_epochs", values_after_flag(stage_a, "--epochs_vae"), ["10000"]),
        ("stage_a_cycles", values_after_flag(stage_a, "--cyclical_beta_n_cycles"), ["125"]),
        ("stage_a_T0", values_after_flag(stage_a, "--lr_scheduler_T0"), ["80"]),
        ("stage_a_patience", values_after_flag(stage_a, "--early_stopping_patience_vae"), ["560"]),
        ("stage_a_metadata_features", values_after_flag(stage_a, "--metadata_features"), ["Age", "Sex"]),
        ("stage_a_norm_mode", values_after_flag(stage_a, "--norm_mode"), ["zscore_offdiag"]),
        ("stage_a_dropout", values_after_flag(stage_a, "--dropout_rate_vae"), ["0.15"]),
        ("stage_a_batch_size", values_after_flag(stage_a, "--batch_size"), ["64"]),
        ("stage_a_sampler_strategy", values_after_flag(stage_a, "--vae_train_sampler_strategy"), ["none"]),
        ("stage_a_seed", values_after_flag(stage_a, "--seed"), ["42"]),
        (
            "stage_a_tensor_is_reprocessed14",
            [values_after_flag(stage_a, "--global_tensor_path")[0]]
            if values_after_flag(stage_a, "--global_tensor_path")
            else [""],
            [EXPECTED_TENSOR_PATH],
        ),
        ("stage_b_model", values_after_flag(stage_b, "--models"), [PRIMARY_MODEL]),
        ("stage_b_outer", values_after_flag(stage_b, "--outer-folds"), ["5"]),
        ("stage_b_inner", values_after_flag(stage_b, "--inner-folds"), ["5"]),
        (
            "oof_run_dir_matches_stage_b",
            values_after_flag(oof, "--run-dir"),
            values_after_flag(stage_b, "--run-dir"),
        ),
    ]
    rows = [
        {
            "check": name,
            "actual": " ".join(actual),
            "expected": " ".join(expected),
            "pass": actual == expected,
        }
        for name, actual, expected in checks
    ]
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError(
            f"Command validation failed for beta={expected_beta}:\n"
            + out[~out["pass"]].to_string(index=False)
        )
    return out


# ── Output writers ────────────────────────────────────────────────────────────

def write_launch_script(
    out_dir: Path,
    stage_a: List[str],
    stage_b: List[str],
    oof: List[str],
    run_id: str,
    beta_str: str,
) -> Path:
    launch_path = out_dir / f"launch_ch1only_beta{beta_str}_reprocessed14.sh"
    lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated launch script — DO NOT RUN without Stage B AUC/PR-AUC/Philips FPR/OASIS checks.",
        f"# Beta sensitivity sweep candidate — post-final exploratory; NOT auto-promoted.",
        f"# Run ID: {run_id}",
        f"# Generated: {now_utc()}",
        "",
        "set -euo pipefail",
        "",
        "# Stage A: VAE + classifier training",
        shlex.join(stage_a),
        "",
        "# Stage B: classifier-only readout (frozen latent)",
        shlex.join(stage_b),
        "",
        "# Stage B OOF score calibration",
        shlex.join(oof),
        "",
        f"echo 'All stages complete for {run_id}'",
    ]
    launch_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    launch_path.chmod(0o755)
    return launch_path


def write_per_candidate_outputs(
    candidate_dir: Path,
    cand: Dict[str, Any],
    config: Dict[str, Any],
    strict_diff: Dict[str, Any],
    diff_vs_ref: pd.DataFrame,
    symlink: Dict[str, Any],
    command_checks: pd.DataFrame,
    stage_a: List[str],
    stage_b: List[str],
    oof: List[str],
) -> Path:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(cand["config"], candidate_dir / "candidate_config.json")
    write_table(diff_vs_ref, candidate_dir, "config_diff_vs_reference")
    write_table(pd.DataFrame([symlink]), candidate_dir, "output_symlink_and_stale_audit")
    write_table(command_checks, candidate_dir, "dryrun_command_validation")
    write_json(candidate_dir / "strict_scientific_diff.json", strict_diff)
    write_json(
        candidate_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "run_id": cand["run_id"],
            "beta": cand["beta"],
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_beta": REFERENCE_BETA,
            "single_scientific_diff": "beta_vae",
            "locked_tensor": EXPECTED_TENSOR_PATH,
            "locked_channels": EXPECTED_CHANNELS,
            "promotion_rule": PROMOTION_RULE,
            "training_launched": False,
            "guardrails": [
                "no training during preflight",
                "no tensor modification",
                "no metadata modification",
                "no threshold refitting",
                "not auto-promoted even if AUC improves",
            ],
        },
    )
    planned = pd.DataFrame([
        {"name": "stage_a_training", "command": shlex.join(stage_a)},
        {"name": "stage_b_classifier_only", "command": shlex.join(stage_b)},
        {"name": "stageB_oof_ecdf_score_harmonization", "command": shlex.join(oof)},
    ])
    write_table(planned, candidate_dir, "planned_commands")
    launch_path = write_launch_script(
        candidate_dir, stage_a, stage_b, oof, cand["run_id"], cand["beta_str"]
    )
    return launch_path


def config_diff_table(reference: Dict[str, Any], target: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for section in ["parameters", "paths", "split_strategy"]:
        src = reference.get(section, {})
        tgt = target.get(section, {})
        for key in sorted(set(src) | set(tgt)):
            old = src.get(key, "<MISSING>")
            new = tgt.get(key, "<MISSING>")
            if old != new:
                rows.append({"section": section, "key": key, "reference_value": old, "target_value": new})
    for key in ["run_name", "description", "selected_channel_names"]:
        if reference.get(key) != target.get(key):
            rows.append({"section": "metadata", "key": key, "reference_value": reference.get(key), "target_value": target.get(key)})
    return pd.DataFrame(rows)


def write_sweep_summary(
    out_dir: Path,
    candidates: List[Dict[str, Any]],
    candidate_results: List[Dict[str, Any]],
) -> None:
    rows = []
    for cand, res in zip(candidates, candidate_results):
        rows.append({
            "beta": cand["beta"],
            "run_id": cand["run_id"],
            "preflight_pass": res["preflight_pass"],
            "output_clean": res["output_clean"],
            "stale_markers": res["stale_markers"],
            "launch_script": res["launch_script"],
        })
    write_table(pd.DataFrame(rows), out_dir, "sweep_summary")


def write_final_recommendation(
    out_dir: Path,
    tensor: Dict[str, Any],
    candidate_results: List[Dict[str, Any]],
) -> None:
    all_pass = all(r["preflight_pass"] for r in candidate_results)
    status = "PASS" if all_pass else "PARTIAL PASS — see per-candidate results"
    path = out_dir / "final_recommendation.md"
    path.write_text(f"""\
# Beta Sensitivity Sweep Preflight — Final Recommendation

**Sweep:** ch1-only Pearson model, Site31 Mayo reprocessed14 tensor
**Reference run:** `{REFERENCE_RUN_ID}` (beta=3.75)
**Candidates:** beta=4.5, beta=5.5
**Date:** {now_utc()}
**Preflight status:** {status}

---

## Context

The reference run (`{REFERENCE_RUN_ID}`) completed with:
- ADNI pooled AUC (OOF ECDF): **0.7877** (PASS vs threshold 0.782951)
- ADNI pooled PR-AUC (OOF ECDF): **0.5697** (PASS vs threshold 0.559873)
- **Philips CN FPR: 0.4949 (49/99)** — CATASTROPHIC FAILURE vs promoted 3-channel reference (8/99 = 0.0808)

This beta sweep tests whether increased disentanglement pressure (beta 3.75 → 4.5 or 5.5)
reduces Philips CN false positives without sacrificing discriminability.

Both candidates use identical setup: channel=[1] (Pearson_Full_FisherZ_Signed),
Site31 Mayo reprocessed14 tensor, all other parameters locked to reference.

---

## What these runs are

**Post-final exploratory beta sensitivity** — NOT automatically promoted even if AUC improves.

The single controlled scientific diff vs reference per candidate: `beta_vae` only.
All other parameters (channels, tensor, latent_dim, T0, epochs, patience, batch, dropout,
recon_loss, norm_mode, seed, folds, classifiers) are locked.

---

## Shared validation (tensor and subject pool — identical for both candidates)

| Check | Result |
|---|---|
| Tensor shape | {tensor["tensor_shape_full"]} ✓ |
| Channel [1] = Pearson_Full_FisherZ_Signed | ✓ |
| N Site31 reprocessed subjects | {tensor["n_site31_mayo_reprocessed_subjects"]} (expected 14) ✓ |
| Stored original SHA256 matches promoted | {tensor["stored_original_sha256_matches_promoted"]} ✓ |
| tensor_idx alignment mismatches | {tensor["tensor_idx_alignment_mismatches"]} ✓ |
| VAE pool | CN=300 / MCI=250 / AD=97 ✓ |
| Classifier pool | CN=300 / AD=97 (N=397) ✓ |
| 035_S_6927 present | ✓ |
| 128_S_2002 absent | ✓ |
| All 5 outer folds feasible | ✓ |

---

## Per-candidate preflight results

| Candidate | beta | Output clean | Preflight pass | Launch script |
|---|---|---|---|---|
{chr(10).join(
    f"| beta={r['beta']} | {r['beta']} | {r['output_clean']} | {r['preflight_pass']} | `{r['launch_script']}` |"
    for r in candidate_results
)}

---

## Promotion criteria (ALL must hold simultaneously)

**1. ADNI Stage B AUC/PR-AUC**
- Pooled AUC > 0.782951 AND PR-AUC >= 0.559873 (locked promotion thresholds)
- Reference: AUC=0.7877, PR-AUC=0.5697 (marginal pass)

**2. OASIS external AUC**
- Must be assessed after training and reported alongside ADNI metrics.
- OASIS AUC must be non-inferior; degradation blocks promotion.

**3. Philips CN FPR non-regression vs promoted 3-channel model**
- Hard gate: pooled Philips CN FPR must be ≤ 0.0808 (promoted reference: 8/99).
- Reference ch1only beta3p75 had FPR=0.4949 (49/99) — improvement required is 6×.
- If FPR does not improve substantially, the sweep is a negative result.

**4. Robustness improvement over ch1only beta3p75 reference**
- AUC improvement alone is insufficient.
- The candidate must show meaningful improvement in scanner-subgroup generalization
  (Philips CN FPR, Site2 FPR, or foldwise AUC variance reduction).

---

## Decision logic after training

If beta=4.5 and beta=5.5 BOTH fail Philips FPR gate:
- This confirms the ch1-only parsimony is fundamentally incompatible with scanner-subgroup
  specificity at this latent capacity and reconstruction objective.
- No further beta-chasing is recommended.
- The 3-channel promoted model remains final.

If one candidate passes all criteria:
- Report the passing candidate's Stage B metrics and Philips FPR against all four gates above.
- Do not promote without confirming OASIS AUC.

---

## Hard constraints compliance

- No model training performed during preflight.
- No tensor modification (original tensor untouched; replacement tensor read-only).
- No metadata modification.
- No threshold refitting.
- No subject exclusion.
- Martín acquisition/QC variables NOT used as features.
""", encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = args.preflight_dir
    python_exe = args.python_executable

    print(f"Beta sweep preflight: {len(CANDIDATES)} candidates")
    print(f"  Reference: {REFERENCE_RUN_ID} (beta={REFERENCE_BETA})")
    for cand in CANDIDATES:
        print(f"  Candidate: {cand['run_id']} (beta={cand['beta']})")
    print(f"  Preflight output: {out_dir}")

    reference_cfg = load_json(REFERENCE_CONFIG)

    # Shared validation (tensor, metadata, pool, folds — identical for all candidates).
    print("\n[SHARED 1/3] Loading and validating metadata / subject pool...")
    meta = load_metadata(reference_cfg)
    pool = validate_subject_pool(meta)

    print("[SHARED 2/3] Validating replacement tensor...")
    tensor = validate_tensor(meta)

    print("[SHARED 3/3] Validating fold feasibility (using reference config; params identical for all candidates)...")
    folds = validate_fold_feasibility(reference_cfg, meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = out_dir / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    write_table(pool, shared_dir, "subject_pool_validation")
    write_table(folds, shared_dir, "fold_feasibility")
    write_table(pd.DataFrame([tensor]), shared_dir, "tensor_validation")
    write_json(shared_dir / "tensor_validation.json", tensor)
    print(f"  Shared outputs written to: {shared_dir}")

    candidate_results: List[Dict[str, Any]] = []
    all_ok = True

    for cand in CANDIDATES:
        beta = cand["beta"]
        beta_str = cand["beta_str"]
        run_id = cand["run_id"]
        print(f"\n[CANDIDATE beta={beta}] {run_id}")

        try:
            target_cfg = load_json(cand["config"])

            print(f"  Validating strict diff (must be only beta_vae: {REFERENCE_BETA} -> {beta})...")
            strict_diff = validate_candidate_diff(reference_cfg, target_cfg, expected_beta=beta)

            diff_table = config_diff_table(reference_cfg, target_cfg)

            print(f"  Checking output symlink / stale markers...")
            symlink = output_symlink_status(target_cfg)

            print(f"  Building and validating commands...")
            stage_a = build_stage_a_command(target_cfg, python_exe)
            stage_b = build_stage_b_command(target_cfg, python_exe)
            oof = build_oof_command(target_cfg, python_exe)
            command_checks = validate_commands(target_cfg, stage_a, stage_b, oof, expected_beta=beta)

            candidate_dir = out_dir / f"beta{beta_str}"
            launch_path = write_per_candidate_outputs(
                candidate_dir=candidate_dir,
                cand=cand,
                config=target_cfg,
                strict_diff=strict_diff,
                diff_vs_ref=diff_table,
                symlink=symlink,
                command_checks=command_checks,
                stage_a=stage_a,
                stage_b=stage_b,
                oof=oof,
            )

            is_clean = symlink["preflight_clean_missing_output_ok"]
            print(f"  Output clean: {is_clean}, stale markers: {symlink['stale_marker_count']}")
            print(f"  Launch script: {launch_path}")
            print(f"  beta={beta}: PASS")

            candidate_results.append({
                "beta": beta,
                "run_id": run_id,
                "preflight_pass": True,
                "output_clean": is_clean,
                "stale_markers": symlink["stale_marker_count"],
                "launch_script": str(launch_path.name),
                "error": "",
            })

        except Exception as exc:
            all_ok = False
            print(f"  FAIL: {exc}", file=sys.stderr)
            candidate_results.append({
                "beta": beta,
                "run_id": run_id,
                "preflight_pass": False,
                "output_clean": False,
                "stale_markers": -1,
                "launch_script": "",
                "error": str(exc),
            })

    write_sweep_summary(out_dir, CANDIDATES, candidate_results)
    write_final_recommendation(out_dir, tensor, candidate_results)

    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "sweep": "ch1only_beta_sweep_reprocessed14",
            "reference_run_id": REFERENCE_RUN_ID,
            "reference_beta": REFERENCE_BETA,
            "candidates": [{"beta": c["beta"], "run_id": c["run_id"], "beta_str": c["beta_str"]} for c in CANDIDATES],
            "tensor_path": EXPECTED_TENSOR_PATH,
            "locked_channels": EXPECTED_CHANNELS,
            "all_candidates_pass": all_ok,
            "candidate_results": candidate_results,
            "promotion_rule": PROMOTION_RULE,
            "guardrails": [
                "no training during preflight",
                "no tensor modification",
                "no metadata modification",
                "no threshold refitting",
                "not auto-promoted even if AUC improves",
            ],
        },
    )

    print("\n--- Sweep summary ---")
    for r in candidate_results:
        status = "PASS" if r["preflight_pass"] else f"FAIL: {r['error']}"
        print(f"  beta={r['beta']}: {status}")

    print(f"\nPreflight outputs: {out_dir}")

    if not all_ok:
        errors = [r for r in candidate_results if not r["preflight_pass"]]
        sys.exit(f"\n{len(errors)} candidate(s) failed preflight.")

    print("\nDry-run OK. All candidate preflights PASS. No training launched.")

    if args.confirm_training:
        target_beta_str = args.confirm_training
        target_cand = next((c for c in CANDIDATES if c["beta_str"] == target_beta_str), None)
        if target_cand is None:
            sys.exit(f"Unknown beta_str '{target_beta_str}'. Valid: {[c['beta_str'] for c in CANDIDATES]}")
        launch = out_dir / f"beta{target_beta_str}" / f"launch_ch1only_beta{target_beta_str}_reprocessed14.sh"
        if not launch.exists():
            sys.exit(f"Launch script not found: {launch}")
        result_for_cand = next(r for r in candidate_results if r["beta"] == target_cand["beta"])
        if not result_for_cand["output_clean"]:
            sys.exit(f"Refusing training for beta={target_cand['beta']}: output dir is not clean.")
        print(f"\nLaunching training for beta={target_cand['beta']}...")
        subprocess.run(["bash", str(launch)], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
