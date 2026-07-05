#!/usr/bin/env python3
"""Capacity-control FULL 5x5 launcher — recover035 latent128 beta2.5 locked schedule.

Base: recover035_full5x5 (patched_metadata_candidate.csv; 035_S_6927 as AD,
      128_S_2002 excluded; latent_dim=256, beta_vae=2.5, epochs_vae=4480,
      cyclical_beta_n_cycles=56, lr_scheduler_T0=80, patience=320)

Controlled change vs recover035_full5x5 (1 parameter diff):
  latent_dim : 256 -> 128

All other parameters unchanged:
  channels_to_use=[1,0,2], beta_vae=2.5, epochs_vae=4480, cyclical_beta_n_cycles=56,
  lr_scheduler_T0=80, early_stopping_patience_vae=320, n_iter_logreg=300,
  n_iter_svm=300, dropout_rate_vae=0.15, batch_size=64, seed=42,
  recon_loss_mode=mse_sum_batchmean_current, norm_mode=zscore_offdiag,
  vae_final_activation=tanh, intermediate_fc_dim_vae=quarter.
  metadata_path=patched_metadata_candidate.csv (same as recover035_full5x5).
  tensor=adni_expanded_v5_1_batch20260514b_no_pybandpass (same).

Stage B readout:
  1. Raw logreg_l2 (via run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py)
  2. OOF-logitz logreg_l2 (via run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py
     pointed at this run's latent cache)

Promotion gate (vs promoted OOF-logitz candidate):
  AUC > 0.7951 AND PR-AUC >= 0.5728 (both simultaneously)
  OR improve Fold 4 / BA / F1 without materially hurting AUC/PR-AUC.

Default behavior is preflight only. Real training requires --confirm-training.
128_S_2002 remains excluded everywhere.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CONFIG = (
    PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_recover035_full5x5.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent128_beta2p5_lockedSchedule_full5x5.json"
)

PATCHED_METADATA = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
EXPECTED_RESCUE_AGE = 59.6
EXPECTED_RESCUE_SEX = "F"
EXPECTED_RESCUE_MANUFACTURER = "SIEMENS"
EXPECTED_RESCUE_DX = "AD"
EXPECTED_RESCUE_TENSOR_IDX = 256

EXPECTED_VAE_N = 647
EXPECTED_VAE_CN = 300
EXPECTED_VAE_MCI = 250
EXPECTED_VAE_AD = 97
EXPECTED_CLF_N = 397
EXPECTED_CLF_CN = 300
EXPECTED_CLF_AD = 97

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_PRIMARY_MODEL = "logreg_l2"
EXPECTED_PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

# Single controlled change vs recover035_full5x5
EXPECTED_LATENT_DIM = 128
BASE_LATENT_DIM = 256
# Locked schedule — unchanged from recover035_full5x5
EXPECTED_EPOCHS_VAE = 4480
EXPECTED_N_CYCLES = 56
EXPECTED_CYCLE_LEN = 80  # 4480 / 56 = 80
EXPECTED_T0 = 80
EXPECTED_PATIENCE = 320  # 4 × 80-epoch cycles
EXPECTED_BETA = 2.5
EXPECTED_N_ITER_LOGREG = 300
EXPECTED_N_ITER_SVM = 300

# Promotion gate: vs promoted OOF-logitz candidate (beta3p75 latent384)
PROMOTED_AUC = 0.7951
PROMOTED_PR_AUC = 0.5728

STAGE_B_RAW_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
)
STAGE_B_OOF_LOGITZ_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"
)

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json", "command_log.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch.")
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-oof-logitz", action="store_true", help="Skip Stage B OOF-logitz step.")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--skip-preview-write", action="store_true")
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help=(
            "Before real training, move existing output_dir contents to a timestamped quarantine "
            "and continue with an empty output_dir. Never deletes silently."
        ),
    )
    return parser.parse_args()


# ── Utilities ────────────────────────────────────────────────────────────────

def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def append_arg(command: List[str], name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(f"--{name}")
        return
    if value is None:
        return
    command.append(f"--{name}")
    if isinstance(value, list):
        command.extend(str(v) for v in value)
    else:
        command.append(str(value))


def values_after_flag(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    out: List[str] = []
    for token in tokens[list(tokens).index(flag) + 1:]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


# ── Config validation ─────────────────────────────────────────────────────────

def assert_latent128_diff(source: Dict[str, Any], target: Dict[str, Any]) -> None:
    """Verify exactly 1 parameter diff vs recover035_full5x5 (latent256).

    Allowed diff:
      latent_dim : 256 -> 128

    Everything else must be identical: channels_to_use, beta_vae, epochs_vae,
    cyclical_beta_n_cycles, lr_scheduler_T0, early_stopping_patience_vae,
    n_iter_logreg, n_iter_svm, tensor path, metadata path.
    """
    src_params = dict(source["parameters"])
    tgt_params = dict(target["parameters"])
    for params in (src_params, tgt_params):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
        params.setdefault("repeated_outer_folds_n_repeats", 1)
    if set(src_params) != set(tgt_params):
        raise RuntimeError(
            "Target config parameter keys differ from source (recover035_full5x5): "
            f"missing={sorted(set(src_params) - set(tgt_params))}, "
            f"extra={sorted(set(tgt_params) - set(src_params))}"
        )
    diffs = {k: (src_params[k], tgt_params[k]) for k in src_params if src_params[k] != tgt_params[k]}
    require_equal(
        diffs,
        {"latent_dim": (BASE_LATENT_DIM, EXPECTED_LATENT_DIM)},
        "scientific parameter diffs (latent128 vs recover035_full5x5)",
    )
    if source["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        raise RuntimeError("global_tensor_path changed; this is not allowed for a capacity-control run.")
    if source["parameters"]["channels_to_use"] != target["parameters"]["channels_to_use"]:
        raise RuntimeError("channels_to_use changed; this is not allowed.")
    src_meta = source["paths"]["metadata_path"]
    tgt_meta = target["paths"]["metadata_path"]
    if not tgt_meta.endswith("patched_metadata_candidate.csv"):
        raise RuntimeError(
            f"metadata_path must end with 'patched_metadata_candidate.csv', got: {tgt_meta!r}"
        )
    if src_meta != tgt_meta:
        raise RuntimeError(
            f"metadata_path changed vs recover035_full5x5 ({src_meta!r} -> {tgt_meta!r}). "
            "latent128 must use the same patched_metadata_candidate.csv."
        )


def validate_config(config: Dict[str, Any], source_config: Dict[str, Any]) -> None:
    assert_latent128_diff(source_config, config)
    params = config["parameters"]
    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY_COLS, "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY_COLS, "vae_stratify_cols")
    require_equal(params["metadata_features"], EXPECTED_METADATA_FEATURES, "metadata_features")
    if "Sex" in params["classifier_stratify_cols"] or "Sex" in params["vae_stratify_cols"]:
        raise RuntimeError("Sex must remain metadata/covariate only.")
    for key, expected in [
        ("outer_folds", 5),
        ("inner_folds", 5),
        ("repeated_outer_folds_n_repeats", 1),
        ("latent_dim", EXPECTED_LATENT_DIM),
        ("epochs_vae", EXPECTED_EPOCHS_VAE),
        ("cyclical_beta_n_cycles", EXPECTED_N_CYCLES),
        ("early_stopping_patience_vae", EXPECTED_PATIENCE),
        ("lr_scheduler_type", "cosine_warm"),
        ("lr_scheduler_T0", EXPECTED_T0),
        ("beta_vae", EXPECTED_BETA),
        ("cyclical_beta_ratio_increase", 0.4),
        ("dropout_rate_vae", 0.15),
        ("vae_dropout_scope", "legacy_all"),
        ("vae_block_order", "legacy_act_norm"),
        ("batch_size", 64),
        ("decoder_type", "convtranspose"),
        ("num_conv_layers_encoder", 4),
        ("norm_mode", "zscore_offdiag"),
        ("recon_loss_mode", "mse_sum_batchmean_current"),
        ("vae_final_activation", "tanh"),
        ("intermediate_fc_dim_vae", "quarter"),
        ("use_layernorm_vae_fc", False),
        ("classifier_calibrate", True),
        ("classifier_use_class_weight", True),
        ("n_iter_logreg", EXPECTED_N_ITER_LOGREG),
        ("n_iter_svm", EXPECTED_N_ITER_SVM),
    ]:
        require_equal(params[key], expected, key)
    require_equal(params.get("vae_train_sampler_strategy", "none"), "none", "vae_train_sampler_strategy")
    actual_cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    if actual_cycle_len != EXPECTED_CYCLE_LEN:
        raise RuntimeError(
            f"Expected {EXPECTED_EPOCHS_VAE}/{EXPECTED_N_CYCLES} = {EXPECTED_CYCLE_LEN} epoch cycle length; "
            f"got {params['epochs_vae']}/{params['cyclical_beta_n_cycles']} = {actual_cycle_len}"
        )
    if params["lr_scheduler_T0"] != EXPECTED_T0:
        raise RuntimeError(
            f"lr_scheduler_T0 must equal {EXPECTED_T0} to remain phase-aligned with beta cycles."
        )
    if actual_cycle_len != params["lr_scheduler_T0"]:
        raise RuntimeError(
            f"Beta cycle length ({actual_cycle_len}) != lr_scheduler_T0 ({params['lr_scheduler_T0']}). "
            "Beta and LR cycles must remain phase-aligned."
        )
    patience = params["early_stopping_patience_vae"]
    if patience % EXPECTED_CYCLE_LEN != 0:
        raise RuntimeError(
            f"early_stopping_patience_vae={patience} is not a multiple of cycle_len={EXPECTED_CYCLE_LEN}."
        )
    n_patience_cycles = patience // EXPECTED_CYCLE_LEN
    if n_patience_cycles != 4:
        raise RuntimeError(
            f"Expected patience = 4 cycle lengths = {4 * EXPECTED_CYCLE_LEN}; "
            f"got {patience} = {n_patience_cycles} cycle lengths."
        )
    for flag in ["qc_analyze_distributions", "qc_check_scanner_leakage", "qc_rate_distortion", "qc_latent_information"]:
        require_equal(params[flag], True, flag)


# ── Dataset inspection ────────────────────────────────────────────────────────

def normalize_manufacturer(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    upper = text.upper()
    if "GE" in upper:
        return "GE"
    if "SIEMENS" in upper:
        return "SIEMENS"
    if "PHILIPS" in upper:
        return "Philips"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str)
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_manufacturer)
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cn_ad["Age"].isna().any() or cn_ad["Sex"].isna().any():
        raise RuntimeError("CN/AD rows contain missing Age/Sex.")
    return meta


def audit_rescue_pool(meta: pd.DataFrame) -> None:
    rg = meta["ResearchGroup_Mapped"]
    cn = int((rg == "CN").sum())
    mci = int((rg == "MCI").sum())
    ad = int((rg == "AD").sum())
    total_vae = len(meta)
    total_clf = cn + ad

    errors: List[str] = []
    if total_vae != EXPECTED_VAE_N:
        errors.append(f"VAE pool n={total_vae}, expected {EXPECTED_VAE_N}")
    if cn != EXPECTED_VAE_CN:
        errors.append(f"VAE CN={cn}, expected {EXPECTED_VAE_CN}")
    if mci != EXPECTED_VAE_MCI:
        errors.append(f"VAE MCI={mci}, expected {EXPECTED_VAE_MCI}")
    if ad != EXPECTED_VAE_AD:
        errors.append(f"VAE AD={ad}, expected {EXPECTED_VAE_AD}")
    if total_clf != EXPECTED_CLF_N:
        errors.append(f"CLF pool n={total_clf}, expected {EXPECTED_CLF_N}")

    row_035 = meta[meta["SubjectID"] == RECOVER_SUBJECT]
    if row_035.empty:
        errors.append(f"{RECOVER_SUBJECT} is NOT in training metadata")
    else:
        r = row_035.iloc[0]
        if abs(float(r["Age"]) - EXPECTED_RESCUE_AGE) > 0.01:
            errors.append(f"{RECOVER_SUBJECT} Age={r['Age']}, expected {EXPECTED_RESCUE_AGE}")
        if str(r["Sex"]) != EXPECTED_RESCUE_SEX:
            errors.append(f"{RECOVER_SUBJECT} Sex={r['Sex']!r}, expected {EXPECTED_RESCUE_SEX!r}")
        if str(r["Manufacturer"]) != EXPECTED_RESCUE_MANUFACTURER:
            errors.append(f"{RECOVER_SUBJECT} Manufacturer={r['Manufacturer']!r}, expected {EXPECTED_RESCUE_MANUFACTURER!r}")
        if str(r["ResearchGroup_Mapped"]) != EXPECTED_RESCUE_DX:
            errors.append(f"{RECOVER_SUBJECT} ResearchGroup_Mapped={r['ResearchGroup_Mapped']!r}, expected {EXPECTED_RESCUE_DX!r}")
        if int(r["tensor_idx"]) != EXPECTED_RESCUE_TENSOR_IDX:
            errors.append(f"{RECOVER_SUBJECT} tensor_idx={r['tensor_idx']}, expected {EXPECTED_RESCUE_TENSOR_IDX}")

    clf_pool = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if (clf_pool["SubjectID"] == EXCLUDED_SUBJECT).any():
        errors.append(f"{EXCLUDED_SUBJECT} appears in classifier pool (should be excluded)")

    if errors:
        raise RuntimeError("Pool audit FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"Pool audit: VAE n={total_vae} (CN={cn}, MCI={mci}, AD={ad}), CLF n={total_clf}  [OK]")
    print(f"Pool audit: {RECOVER_SUBJECT} present, Age={EXPECTED_RESCUE_AGE}, "
          f"Sex={EXPECTED_RESCUE_SEX}, Manufacturer={EXPECTED_RESCUE_MANUFACTURER}  [OK]")
    print(f"Pool audit: {EXCLUDED_SUBJECT} absent from classifier pool  [OK]")


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        subject_ids = zf["subject_ids"].astype(str).tolist()
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "tensor selected channel names")
    if RECOVER_SUBJECT not in subject_ids:
        raise RuntimeError(f"{RECOVER_SUBJECT} not found in tensor subject_ids")
    actual_idx = subject_ids.index(RECOVER_SUBJECT)
    if actual_idx != EXPECTED_RESCUE_TENSOR_IDX:
        raise RuntimeError(
            f"{RECOVER_SUBJECT} at tensor index {actual_idx}, expected {EXPECTED_RESCUE_TENSOR_IDX}"
        )
    return {
        "shape": shape,
        "selected_channel_names": selected,
        "python_bandpass_applied": False,
        "n_subjects": len(subject_ids),
    }


# ── Split preview ─────────────────────────────────────────────────────────────

def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_fields(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["AD", "CN", "MCI"]:
        out[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        out[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
    return out


def make_split_preview(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    seed = int(params["seed"])
    n_splits = int(params["outer_folds"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    strat_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    y_outer = strat_key(cn_ad, strat_cols)
    if (y_outer.value_counts() < n_splits).any():
        raise RuntimeError("Manufacturer-aware outer 5-fold split is not feasible.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    rows: List[Dict[str, Any]] = []
    subjects: List[Dict[str, Any]] = []
    for fold, (train_dev_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), y_outer), start=1):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        if (vae_pool["SubjectID"] == EXCLUDED_SUBJECT).any():
            raise RuntimeError(
                f"Fold {fold}: {EXCLUDED_SUBJECT} is present in the VAE pool. "
                "This subject must be absent from the training-ready metadata."
            )
        vae_cols = ["ResearchGroup_Mapped", *params["vae_stratify_cols"]]
        vae_key = strat_key(vae_pool, vae_cols)
        if (vae_key.value_counts() < 2).any():
            raise RuntimeError("Manufacturer-aware VAE internal validation split is not feasible.")
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + fold + 9,
            shuffle=True,
        )
        components = [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_pool.iloc[train_local]),
            ("vae_internal_val", vae_pool.iloc[val_local]),
        ]
        for component, df in components:
            row: Dict[str, Any] = {
                "fold": fold,
                "split_component": component,
            }
            row.update(count_fields(df))
            rows.append(row)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subjects.append({
                    "fold": fold,
                    "split_component": split_name,
                    "SubjectID": r["SubjectID"],
                    "tensor_idx": int(r["tensor_idx"]),
                    "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                    "Manufacturer": r["Manufacturer"],
                })
    return pd.DataFrame(rows), pd.DataFrame(subjects)


# ── Command builders ───────────────────────────────────────────────────────────

def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path", str(resolve(paths["global_tensor_path"])),
        "--metadata_path", str(resolve(paths["metadata_path"])),
        "--output_dir", str(resolve(paths["output_dir"])),
    ]
    for name, value in params.items():
        append_arg(command, name, value)
    command.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    command.append("--vae_abort_if_val_split_fails")
    _validate_stage_a_command(command)
    return command


def build_stage_b_raw_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    command = [
        python_exe,
        str(STAGE_B_RAW_SCRIPT),
        "--run-dir", str(outdir),
        "--output-dir", str(outdir / "classifier_only_readout"),
        "--outer-folds", str(params["outer_folds"]),
        "--inner-folds", str(params["inner_folds"]),
        "--models", EXPECTED_PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]
    _validate_stage_b_raw_command(command)
    return command


def build_stage_b_oof_logitz_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    oof_logitz_outdir = outdir.parent / f"{outdir.name}_stageB_oof_logitz"
    return [
        python_exe,
        str(STAGE_B_OOF_LOGITZ_SCRIPT),
        "--run-dir", str(outdir),
        "--output-dir", str(oof_logitz_outdir),
    ]


def _validate_stage_a_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--channels_to_use"), ["1", "0", "2"], "Stage A channels")
    require_equal(values_after_flag(command, "--outer_folds"), ["5"], "Stage A outer_folds")
    require_equal(values_after_flag(command, "--inner_folds"), ["5"], "Stage A inner_folds")
    require_equal(values_after_flag(command, "--latent_dim"), [str(EXPECTED_LATENT_DIM)], "Stage A latent_dim")
    require_equal(values_after_flag(command, "--epochs_vae"), [str(EXPECTED_EPOCHS_VAE)], "Stage A epochs_vae")
    require_equal(values_after_flag(command, "--cyclical_beta_n_cycles"), [str(EXPECTED_N_CYCLES)], "Stage A cycles")
    require_equal(values_after_flag(command, "--lr_scheduler_T0"), [str(EXPECTED_T0)], "Stage A T0")
    require_equal(values_after_flag(command, "--beta_vae"), [str(EXPECTED_BETA)], "Stage A beta_vae")
    require_equal(values_after_flag(command, "--early_stopping_patience_vae"), [str(EXPECTED_PATIENCE)], "Stage A patience")
    meta_path = values_after_flag(command, "--metadata_path")
    if not meta_path or not meta_path[0].endswith("patched_metadata_candidate.csv"):
        raise RuntimeError(f"Stage A --metadata_path must end with patched_metadata_candidate.csv, got: {meta_path}")
    req_meta = values_after_flag(command, "--vae_required_metadata_cols")
    require_equal(req_meta, ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"], "Stage A --vae_required_metadata_cols")
    if "--vae_abort_if_val_split_fails" not in command:
        raise RuntimeError("Stage A command missing --vae_abort_if_val_split_fails.")
    sampler_vals = values_after_flag(command, "--vae_train_sampler_strategy")
    if sampler_vals and sampler_vals[0] != "none":
        raise RuntimeError(f"vae_train_sampler_strategy must be 'none', got {sampler_vals[0]!r}")


def _validate_stage_b_raw_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(command, "--inner-folds"), ["5"], "Stage B inner-folds")
    require_equal(values_after_flag(command, "--models"), [EXPECTED_PRIMARY_MODEL], "Stage B models")


# ── Output management ─────────────────────────────────────────────────────────

def stale_output_markers(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    markers: List[Path] = []
    for child in output_dir.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in output_dir.rglob("latent_cache"):
        if nested not in markers:
            markers.append(nested)
    return sorted(markers, key=lambda p: str(p))


def quarantine_existing_output_contents(output_dir: Path) -> Path:
    if not output_dir.exists():
        raise RuntimeError(f"Cannot quarantine missing output_dir: {output_dir}")
    real_output = output_dir.resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = real_output.parent / f"{real_output.name}_quarantine_{timestamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = real_output.parent / f"{real_output.name}_quarantine_{timestamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in list(output_dir.iterdir()):
        child.rename(quarantine / child.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    return quarantine


def ensure_output_prepared(config: Dict[str, Any], force_clean: bool = False) -> Path | None:
    output_dir = resolve(config["paths"]["output_dir"])
    big_disk = Path(config["paths"]["big_disk_output_dir"])
    if not output_dir.exists():
        raise RuntimeError(
            "Refusing to start training: output_dir is missing. Create the external target/symlink first:\n"
            f"  mkdir -p {shlex.quote(str(big_disk))}\n"
            f"  ln -s {shlex.quote(str(big_disk))} {shlex.quote(str(output_dir))}"
        )
    if not output_dir.is_symlink():
        raise RuntimeError(f"Refusing to start training: output_dir is not a symlink: {output_dir}")
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(
            f"Refusing to start training: symlink target is {output_dir.resolve()}, "
            f"expected {big_disk.resolve()}"
        )
    stale = stale_output_markers(output_dir)
    if stale and not force_clean:
        preview = "\n".join(f"  - {p}" for p in stale[:20])
        raise RuntimeError(
            "Refusing to start training: output_dir contains stale run artifacts. "
            "Pass --force-clean to quarantine existing contents before training.\n"
            f"{preview}"
        )
    quarantine: Path | None = None
    if force_clean:
        existing = list(output_dir.iterdir())
        if existing:
            quarantine = quarantine_existing_output_contents(output_dir)
            print(f"[clean-run] Existing output_dir contents moved to quarantine: {quarantine}")
    unexpected = [
        p.name for p in output_dir.iterdir()
        if p.name not in {"run_manifest.json", "command_log.json", "training_stdout.log"}
    ]
    if unexpected:
        raise RuntimeError(
            f"Refusing to start training: output_dir is not empty after clean policy: {unexpected[:8]}"
        )
    return quarantine


def verify_fresh_vae_checkpoints(config: Dict[str, Any], run_start_epoch: float) -> List[Dict[str, Any]]:
    output_dir = resolve(config["paths"]["output_dir"])
    rows: List[Dict[str, Any]] = []
    for fold in range(1, int(config["parameters"]["outer_folds"]) + 1):
        ckpt = output_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        exists = ckpt.exists()
        mtime = ckpt.stat().st_mtime if exists else None
        rows.append({
            "fold": fold,
            "path": str(ckpt),
            "exists": bool(exists),
            "mtime_epoch": mtime,
            "mtime_iso": datetime.fromtimestamp(mtime).isoformat() if mtime is not None else "",
            "fresh_after_run_start": bool(exists and mtime is not None and mtime > run_start_epoch),
        })
    stale = [row for row in rows if not row["fresh_after_run_start"]]
    if stale:
        details = "\n".join(
            f"  - fold {row['fold']}: exists={row['exists']} mtime={row['mtime_iso']}"
            for row in stale
        )
        raise RuntimeError(
            "Refusing to run Stage B: not all VAE checkpoints were created after this launcher invocation.\n"
            f"{details}"
        )
    return rows


# ── Manifest / log ────────────────────────────────────────────────────────────

def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    stage_a: Sequence[str],
    stage_b_raw: Sequence[str],
    stage_b_oof: Sequence[str],
    quarantine_dir: Path | None = None,
) -> Path:
    output_dir = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "controlled_change": (
            "Base: recover035_full5x5 (patched_metadata_candidate.csv, 035_S_6927 as AD, "
            "128_S_2002 excluded, latent_dim=256). "
            f"Single diff: latent_dim {BASE_LATENT_DIM}->{EXPECTED_LATENT_DIM}. "
            "All other parameters unchanged: beta_vae=2.5, epochs_vae=4480, n_cycles=56, "
            "cycle_len=80, T0=80, patience=320 (4 cycles), channels=[1,0,2]."
        ),
        "latent128_diff_vs_recover035_full5x5_verified": True,
        "channels_to_use": EXPECTED_CHANNELS,
        "selected_channel_names": EXPECTED_SELECTED_NAMES,
        "python_bandpass_applied": False,
        "recover_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDED_SUBJECT,
        "expected_vae_pool": EXPECTED_VAE_N,
        "expected_clf_pool": EXPECTED_CLF_N,
        "latent_dim": EXPECTED_LATENT_DIM,
        "base_latent_dim": BASE_LATENT_DIM,
        "beta_vae": EXPECTED_BETA,
        "epochs_vae": EXPECTED_EPOCHS_VAE,
        "cyclical_beta_n_cycles": EXPECTED_N_CYCLES,
        "cycle_len_epochs": EXPECTED_CYCLE_LEN,
        "lr_scheduler_T0": EXPECTED_T0,
        "early_stopping_patience_vae": EXPECTED_PATIENCE,
        "patience_in_cycles": EXPECTED_PATIENCE // EXPECTED_CYCLE_LEN,
        "stage_b_c_grid": "original [1e-3, 1e-2, 1e-1, 1.0]",
        "promotion_rule": (
            f"Primary: AUC > {PROMOTED_AUC} AND PR-AUC >= {PROMOTED_PR_AUC} (both simultaneously; "
            "vs promoted OOF-logitz candidate beta3p75 latent384). "
            "Alternative: improve Fold 4 / BA / F1 without materially hurting AUC/PR-AUC."
        ),
        "stage_a_command": list(stage_a),
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_raw_command": list(stage_b_raw),
        "stage_b_raw_command_shell": shlex.join(stage_b_raw),
        "stage_b_oof_logitz_command": list(stage_b_oof),
        "stage_b_oof_logitz_command_shell": shlex.join(stage_b_oof),
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else "",
        "ranking_source": "Stage B classifier-only logreg_l2 + OOF-logitz calibration",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_command_log(
    config: Dict[str, Any],
    stage_a_rc: int | None,
    stage_b_raw_rc: int | None,
    stage_b_oof_rc: int | None,
    quarantine_dir: Path | None = None,
) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "training_launched": stage_a_rc is not None,
        "stage_a_returncode": stage_a_rc,
        "stage_b_raw_returncode": stage_b_raw_rc,
        "stage_b_oof_logitz_returncode": stage_b_oof_rc,
        "controlled_change": (
            "Base: recover035_full5x5. "
            f"Single diff: latent_dim {BASE_LATENT_DIM}->{EXPECTED_LATENT_DIM}. "
            "Schedule, beta_vae, channels, loss, dropout, batch_size, n_iter unchanged."
        ),
        "recover_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDED_SUBJECT,
        "promotion_rule": (
            f"Primary: AUC > {PROMOTED_AUC} AND PR-AUC >= {PROMOTED_PR_AUC} (vs promoted candidate). "
            "Alternative: improve Fold 4/BA/F1 without materially hurting AUC/PR-AUC."
        ),
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else "",
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
        "promoted_candidate_outputs_modified": False,
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    config = load_json(args.config)
    source = load_json(args.source_config)
    validate_config(config, source)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    if not args.dry_run and not args.confirm_training:
        raise SystemExit(
            "Refusing to launch training without --confirm-training. "
            "Use --dry-run for preflight only."
        )

    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    meta = load_metadata(resolve(config["paths"]["metadata_path"]))
    audit_rescue_pool(meta)
    preview_summary, preview_subjects = make_split_preview(meta, config)

    recover_in_test = preview_subjects[
        (preview_subjects["SubjectID"] == RECOVER_SUBJECT)
        & (preview_subjects["split_component"] == "classifier_test")
    ]
    recover_fold = int(recover_in_test["fold"].iloc[0]) if not recover_in_test.empty else None

    if not args.skip_preview_write:
        preview_csv = resolve(config["paths"]["split_preview_csv"])
        preview_summary_csv = resolve(config["paths"]["split_preview_summary_csv"])
        preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview_subjects.to_csv(preview_csv, index=False)
        preview_summary.to_csv(preview_summary_csv, index=False)

    params = config["parameters"]
    stage_a = build_stage_a_command(config, python_exe)
    stage_b_raw = build_stage_b_raw_command(config, python_exe)
    stage_b_oof = build_stage_b_oof_logitz_command(config, python_exe)
    output_dir = resolve(config["paths"]["output_dir"])
    stale = stale_output_markers(output_dir)
    mode = "DRY-RUN" if args.dry_run else "REAL RUN (CONFIRMED)"
    cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]

    print(f"Config           : {args.config}")
    print(f"Source config    : {args.source_config} (recover035_full5x5 latent256)")
    print(f"Run name         : {config['run_name']}")
    print(f"Mode             : {mode}")
    print(f"Controlled diff  : 1 parameter diff vs recover035_full5x5 (latent_dim {BASE_LATENT_DIM}->{EXPECTED_LATENT_DIM})")
    print(f"Metadata         : {resolve(config['paths']['metadata_path'])}")
    print(f"Tensor           : {config['paths']['global_tensor_path']}")
    print(f"  shape={tensor_info['shape']}, n_subjects={tensor_info['n_subjects']}, "
          f"python_bandpass_applied={tensor_info['python_bandpass_applied']}")
    print(f"Recovered subj   : {RECOVER_SUBJECT} (AD, Age={EXPECTED_RESCUE_AGE}, "
          f"Sex={EXPECTED_RESCUE_SEX}, Manufacturer={EXPECTED_RESCUE_MANUFACTURER}, "
          f"tensor_idx={EXPECTED_RESCUE_TENSOR_IDX})")
    print(f"Excluded subj    : {EXCLUDED_SUBJECT} (remains excluded)")
    print(f"latent_dim       : {BASE_LATENT_DIM} -> {EXPECTED_LATENT_DIM} (only change vs recover035_full5x5)")
    print(f"beta_vae         : {EXPECTED_BETA} (UNCHANGED)")
    print(f"Epochs/cycles    : {params['epochs_vae']}/{params['cyclical_beta_n_cycles']} = {cycle_len:.0f} epoch cycle (UNCHANGED)")
    print(f"lr_scheduler_T0  : {params['lr_scheduler_T0']} (UNCHANGED — phase-aligned)")
    print(f"Patience         : {params['early_stopping_patience_vae']} = "
          f"{params['early_stopping_patience_vae'] // EXPECTED_CYCLE_LEN} × {EXPECTED_CYCLE_LEN}-epoch cycles (UNCHANGED)")
    print(f"n_iter_logreg    : {params['n_iter_logreg']} (UNCHANGED)")
    print(f"Channels         : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_SELECTED_NAMES)})")
    print("Recon loss       : mse_sum_batchmean_current (unchanged)")
    print("Python bandpass  : OFF")
    print(f"VAE pool         : n={EXPECTED_VAE_N} (CN={EXPECTED_VAE_CN}, MCI={EXPECTED_VAE_MCI}, AD={EXPECTED_VAE_AD})")
    print(f"CLF pool         : n={EXPECTED_CLF_N} (CN={EXPECTED_CLF_CN}, AD={EXPECTED_CLF_AD})")
    print(f"035_S_6927 outer-test fold: {recover_fold}")
    print(f"Promotion rule   : AUC > {PROMOTED_AUC} AND PR-AUC >= {PROMOTED_PR_AUC} (vs promoted beta3p75 latent384 OOF-logitz)")
    print(f"                   OR improve Fold 4/BA/F1 without materially hurting AUC/PR-AUC.")
    preview_cols = ["fold", "split_component", "n", "AD", "CN", "MCI",
                    "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips"]
    print(preview_summary[[c for c in preview_cols if c in preview_summary.columns]].to_string(index=False))

    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B raw command:")
    print(shlex.join(stage_b_raw))
    print("\nStage B OOF-logitz command:")
    print(shlex.join(stage_b_oof))
    print("\nClean-run policy:")
    print("Real training refuses stale fold_*, classifier_only_readout, latent_cache, "
          "all_folds_metrics, summary_metrics, run_manifest.json unless --force-clean is passed.")
    if stale:
        print(f"Detected stale markers in output_dir: {len(stale)}")
        for marker in stale[:20]:
            print(f"  - {marker}")
        if args.dry_run:
            print("Dry-run only: no quarantine or cleanup was performed.")
    else:
        print("No stale markers detected in output_dir.")
    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    quarantine_dir = ensure_output_prepared(config, force_clean=args.force_clean)
    run_start_epoch = time.time()
    manifest = write_manifest(args.config, config, stage_a, stage_b_raw, stage_b_oof, quarantine_dir)
    print(f"\nRun manifest written: {manifest}")

    completed = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, None, None, quarantine_dir)
    if completed.returncode != 0:
        return int(completed.returncode)

    fresh_rows = verify_fresh_vae_checkpoints(config, run_start_epoch)
    print("[fresh-checkpoint] All fold_1..fold_5 VAE checkpoints exist and are newer than run start.")
    for row in fresh_rows:
        print(f"  - fold {row['fold']}: {row['mtime_iso']}")

    if args.skip_classifier_readout:
        print("Stage A completed. Stage B skipped (--skip-classifier-readout).")
        return 0

    raw_completed = subprocess.run(stage_b_raw, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, raw_completed.returncode, None, quarantine_dir)
    if raw_completed.returncode != 0:
        return int(raw_completed.returncode)
    print("\nStage B raw completed.")

    if args.skip_oof_logitz:
        print("Stage B OOF-logitz skipped (--skip-oof-logitz).")
        write_command_log(config, completed.returncode, raw_completed.returncode, None, quarantine_dir)
        return 0

    oof_completed = subprocess.run(stage_b_oof, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, raw_completed.returncode, oof_completed.returncode, quarantine_dir)
    if oof_completed.returncode != 0:
        print(f"\nStage B OOF-logitz returned non-zero: {oof_completed.returncode}")
        return int(oof_completed.returncode)
    print("\nStage B OOF-logitz completed.")
    print(f"OOF-logitz output: {resolve(config['paths']['output_dir']).parent / (resolve(config['paths']['output_dir']).name + '_stageB_oof_logitz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
