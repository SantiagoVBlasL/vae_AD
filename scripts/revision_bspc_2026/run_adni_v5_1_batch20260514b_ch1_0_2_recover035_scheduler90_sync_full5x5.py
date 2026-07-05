#!/usr/bin/env python3
"""Prepare/run the controlled exploratory recover035_scheduler90_sync FULL 5x5.

Two controlled changes vs locked v5.1b [1,0,2] horizon4480/cycles56:
  1. Metadata pool rescue: metadata_path -> patched_metadata_candidate.csv
     Adds 035_S_6927 (AD, Age=59.6, Sex=F, Manufacturer=SIEMENS, tensor_index=256)
     to the classifier pool (n=396->397, AD=96->97) and VAE pool (n=646->647, AD=96->97).
  2. Scheduler synchronization:
     epochs_vae          : 4480 -> 4500
     cyclical_beta_n_cycles: 56  -> 50
     lr_scheduler_T0     : 80   -> 90
     Both β and LR cosine warm-restart cycles are now exactly 90 epochs each.
     Phase-aligned restarts at epochs 0, 90, 180, ..., 4410.

128_S_2002 remains excluded. Tensor n=648; patched metadata n=647 (128_S_2002 absent).
Stage A injects --vae_required_metadata_cols and --vae_abort_if_val_split_fails to prevent
the tensor/metadata mismatch failure (128_S_2002 gets a NaN metadata row via LEFT JOIN).

Default behavior is preflight only. Real training requires --confirm-training.
Promotion rule: candidate must beat BOTH AUC > 0.782951 AND PR-AUC >= 0.559873
simultaneously (locked v5.1b horizon4480/cycles56 FULL [1,0,2] reference values).
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
    PROJECT_ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_recover035_scheduler90_sync_full5x5.json"
)

PATCHED_METADATA = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)
ORIGINAL_METADATA_SUFFIX = "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"

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

EXPECTED_CYCLE_LEN = 90
EXPECTED_EPOCHS_VAE = 4500
EXPECTED_N_CYCLES = 50
EXPECTED_T0 = 90

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873

REQUIRED_METADATA_COLS = ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]

STAGE_B_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
)
COMPARISON_SCRIPT = (
    PROJECT_ROOT
    / "scripts/revision_bspc_2026/compare_v5_1_batch20260514b_recover035_scheduler90_sync_full5x5.py"
)
INTEGRITY_AUDIT_SCRIPT = (
    PROJECT_ROOT
    / "scripts/revision_bspc_2026/audit_v5_1_batch20260514b_recover035_scheduler90_sync_integrity.py"
)
POOL_AUDIT_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/audit_adni_recover035_scheduler90_sync_subject_pool.py"
)
SCHEDULE_PHASE_SCRIPT = (
    PROJECT_ROOT / "scripts/revision_bspc_2026/audit_scheduler_cycle90_phase_preflight.py"
)

OUT_COMPARISON_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/recover035_scheduler90_sync_full5x5_comparison"
)

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-config", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument(
        "--confirm-training", action="store_true", help="Required for real Stage A/Stage B launch."
    )
    parser.add_argument("--skip-classifier-readout", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--skip-preview-write", action="store_true")
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help=(
            "Before real training, move existing output_dir contents to a timestamped quarantine folder "
            "and continue with an empty output_dir. Never deletes silently."
        ),
    )
    return parser.parse_args()


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
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def assert_recover035_scheduler90_sync_diff(
    source: Dict[str, Any], target: Dict[str, Any]
) -> None:
    """Verify exactly 3 scheduler parameter diffs vs locked v5.1b PLUS metadata_path change.

    Parameter diffs must be exactly:
      epochs_vae: 4480 -> 4500
      cyclical_beta_n_cycles: 56 -> 50
      lr_scheduler_T0: 80 -> 90

    Non-parameter changes:
      metadata_path: locked CSV -> patched_metadata_candidate.csv
      global_tensor_path: unchanged (enforced)
      channels_to_use: unchanged (enforced)
    """
    src_params = dict(source["parameters"])
    tgt_params = dict(target["parameters"])
    for params in (src_params, tgt_params):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
    if set(src_params) != set(tgt_params):
        raise RuntimeError(
            "Target config parameter keys differ from locked source config: "
            f"missing={sorted(set(src_params) - set(tgt_params))}, "
            f"extra={sorted(set(tgt_params) - set(src_params))}"
        )
    diffs = {k: (src_params[k], tgt_params[k]) for k in src_params if src_params[k] != tgt_params[k]}
    require_equal(
        diffs,
        {
            "epochs_vae": (4480, 4500),
            "cyclical_beta_n_cycles": (56, 50),
            "lr_scheduler_T0": (80, 90),
        },
        "scientific parameter diffs (recover035_scheduler90_sync)",
    )
    if source["paths"]["global_tensor_path"] != target["paths"]["global_tensor_path"]:
        raise RuntimeError("global_tensor_path changed; this is not allowed.")
    if source["parameters"]["channels_to_use"] != target["parameters"]["channels_to_use"]:
        raise RuntimeError("channels_to_use changed; this is not allowed.")
    # Metadata path must have changed from locked to patched
    src_meta = source["paths"]["metadata_path"]
    tgt_meta = target["paths"]["metadata_path"]
    if src_meta == tgt_meta:
        raise RuntimeError(
            "metadata_path is unchanged from locked source; expected it to point to "
            "patched_metadata_candidate.csv"
        )
    if not tgt_meta.endswith("patched_metadata_candidate.csv"):
        raise RuntimeError(
            f"metadata_path must end with 'patched_metadata_candidate.csv', got: {tgt_meta!r}"
        )
    if ORIGINAL_METADATA_SUFFIX in tgt_meta:
        raise RuntimeError(
            "metadata_path still points to original training_ready_metadata; "
            "expected patched_metadata_candidate.csv"
        )


def validate_config(config: Dict[str, Any], source_config: Dict[str, Any]) -> None:
    assert_recover035_scheduler90_sync_diff(source_config, config)
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
        ("latent_dim", 256),
        ("epochs_vae", EXPECTED_EPOCHS_VAE),
        ("cyclical_beta_n_cycles", EXPECTED_N_CYCLES),
        ("early_stopping_patience_vae", 320),
        ("lr_scheduler_type", "cosine_warm"),
        ("lr_scheduler_T0", EXPECTED_T0),
        ("beta_vae", 2.5),
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
        ("n_iter_logreg", 300),
        ("n_iter_svm", 300),
    ]:
        require_equal(params[key], expected, key)
    require_equal(
        params.get("vae_train_sampler_strategy", "none"), "none", "vae_train_sampler_strategy"
    )
    # Core synchronization invariant
    actual_cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    if actual_cycle_len != EXPECTED_CYCLE_LEN:
        raise RuntimeError(
            f"Expected {EXPECTED_EPOCHS_VAE}/{EXPECTED_N_CYCLES} = {EXPECTED_CYCLE_LEN} epoch cycle length; "
            f"got {params['epochs_vae']}/{params['cyclical_beta_n_cycles']} = {actual_cycle_len}"
        )
    if params["lr_scheduler_T0"] != EXPECTED_T0:
        raise RuntimeError(f"lr_scheduler_T0 must equal {EXPECTED_T0} to synchronize with beta cycle.")
    if actual_cycle_len != params["lr_scheduler_T0"]:
        raise RuntimeError(
            f"Beta cycle length ({actual_cycle_len}) must equal lr_scheduler_T0 ({params['lr_scheduler_T0']}) "
            "for synchronized scheduling."
        )
    for flag in [
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
    ]:
        require_equal(params[flag], True, flag)


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
    """Verify 035 is present with correct fields, 128 is absent; check pool sizes."""
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
            errors.append(
                f"{RECOVER_SUBJECT} Manufacturer={r['Manufacturer']!r}, "
                f"expected {EXPECTED_RESCUE_MANUFACTURER!r}"
            )
        if str(r["ResearchGroup_Mapped"]) != EXPECTED_RESCUE_DX:
            errors.append(
                f"{RECOVER_SUBJECT} ResearchGroup_Mapped={r['ResearchGroup_Mapped']!r}, "
                f"expected {EXPECTED_RESCUE_DX!r}"
            )
        if int(r["tensor_idx"]) != EXPECTED_RESCUE_TENSOR_IDX:
            errors.append(
                f"{RECOVER_SUBJECT} tensor_idx={r['tensor_idx']}, expected {EXPECTED_RESCUE_TENSOR_IDX}"
            )

    clf_pool = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if (clf_pool["SubjectID"] == EXCLUDED_SUBJECT).any():
        errors.append(f"{EXCLUDED_SUBJECT} appears in classifier pool (should be excluded)")

    if errors:
        raise RuntimeError("Pool audit FAILED:\n" + "\n".join(f"  - {e}" for e in errors))

    print(
        f"Pool audit: VAE n={total_vae} (CN={cn}, MCI={mci}, AD={ad}), "
        f"CLF n={total_clf} (CN={cn}, AD={ad})  [OK]"
    )
    print(
        f"Pool audit: {RECOVER_SUBJECT} present, Age={EXPECTED_RESCUE_AGE}, "
        f"Sex={EXPECTED_RESCUE_SEX}, Manufacturer={EXPECTED_RESCUE_MANUFACTURER}  [OK]"
    )
    print(f"Pool audit: {EXCLUDED_SUBJECT} absent from classifier pool  [OK]")


def inspect_tensor(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as zf:
        if bool(zf["python_bandpass_applied"]):
            raise RuntimeError("Refusing tensor with python_bandpass_applied=True")
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "tensor selected channel names")
    with np.load(path, allow_pickle=False) as zf:
        subject_ids = zf["subject_ids"].astype(str).tolist()
    if RECOVER_SUBJECT not in subject_ids:
        raise RuntimeError(f"{RECOVER_SUBJECT} not found in tensor subject_ids")
    actual_idx = subject_ids.index(RECOVER_SUBJECT)
    if actual_idx != EXPECTED_RESCUE_TENSOR_IDX:
        raise RuntimeError(
            f"{RECOVER_SUBJECT} at tensor index {actual_idx}, expected {EXPECTED_RESCUE_TENSOR_IDX}"
        )
    print(f"Tensor: {RECOVER_SUBJECT} confirmed at tensor_index={actual_idx}  [OK]")
    return {
        "shape": shape,
        "selected_channel_names": selected,
        "python_bandpass_applied": False,
        "n_subjects": len(subject_ids),
    }


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
        for dx in ["AD", "CN", "MCI"]:
            out[f"{dx}_{label}"] = int(
                (df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(mfr)).sum()
            )
    for sex in ["F", "M"]:
        out[f"Sex_{sex}"] = int(df["Sex"].eq(sex).sum())
    out[f"contains_{RECOVER_SUBJECT.replace('-', '_')}"] = bool(
        (df["SubjectID"] == RECOVER_SUBJECT).any()
    )
    return out


def validate_component_counts(row: Dict[str, Any], component: str) -> List[str]:
    errors: List[str] = []
    diagnoses = ["AD", "CN"] if component in {"classifier_train_dev", "classifier_test"} else ["AD", "CN", "MCI"]
    for dx in diagnoses:
        if row[dx] <= 0:
            errors.append(f"{component} missing {dx}")
    for mfr in ["GE", "Siemens", "Philips"]:
        if row[f"Manufacturer_{mfr}"] <= 0:
            errors.append(f"{component} missing Manufacturer {mfr}")
    return errors


def make_split_preview(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Guard: EXCLUDED_SUBJECT must never appear in the patched metadata
    if EXCLUDED_SUBJECT in meta["SubjectID"].values:
        raise RuntimeError(
            f"POOL GUARD: {EXCLUDED_SUBJECT} is present in patched metadata. "
            "This subject must remain absent. Check that metadata_path points to "
            "patched_metadata_candidate.csv."
        )
    # Verify RECOVER_SUBJECT is present
    if RECOVER_SUBJECT not in meta["SubjectID"].values:
        raise RuntimeError(
            f"POOL GUARD: {RECOVER_SUBJECT} is absent from patched metadata. "
            "Check that metadata_path points to patched_metadata_candidate.csv."
        )
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
    for fold, (train_dev_idx, test_idx) in enumerate(
        splitter.split(np.zeros(len(cn_ad)), y_outer), start=1
    ):
        train_dev = cn_ad.iloc[train_dev_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(
            all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False
        )
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        # Per-fold guard: 128_S_2002 must not appear (it's absent from patched metadata)
        if (vae_pool["SubjectID"] == EXCLUDED_SUBJECT).any():
            raise RuntimeError(
                f"Fold {fold}: {EXCLUDED_SUBJECT} is present in the VAE pool from the patched metadata. "
                "This subject must be absent from the training-ready pool."
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
                "classifier_stratification_cols": "+".join(strat_cols),
                "vae_internal_val_stratification_cols": "+".join(vae_cols),
            }
            row.update(count_fields(df))
            errors = validate_component_counts(row, component)
            row["passes_required_representation_check"] = not errors
            row["representation_check_errors"] = " | ".join(errors)
            rows.append(row)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subjects.append(
                    {
                        "fold": fold,
                        "split_component": split_name,
                        "SubjectID": r["SubjectID"],
                        "tensor_idx": int(r["tensor_idx"]),
                        "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                        "Manufacturer": r["Manufacturer"],
                        "Sex": r["Sex"],
                        f"is_{RECOVER_SUBJECT.replace('-', '_')}": bool(
                            r["SubjectID"] == RECOVER_SUBJECT
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(subjects)


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    command = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for name, value in params.items():
        append_arg(command, name, value)
    # Metadata pool guards: filter NaN-metadata subjects from VAE pool, abort on split failure
    command.append("--vae_required_metadata_cols")
    command.extend(REQUIRED_METADATA_COLS)
    command.append("--vae_abort_if_val_split_fails")
    validate_stage_a_command(command)
    return command


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    params = config["parameters"]
    outdir = resolve(paths["output_dir"])
    command = [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(outdir),
        "--output-dir",
        str(outdir / "classifier_only_readout"),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        EXPECTED_PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]
    validate_stage_b_command(command)
    return command


def build_comparison_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(COMPARISON_SCRIPT),
        "--candidate-run-dir",
        str(outdir),
        "--candidate-readout-dir",
        str(outdir / "classifier_only_readout"),
        "--output-dir",
        str(OUT_COMPARISON_DIR),
    ]


def build_integrity_audit_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(INTEGRITY_AUDIT_SCRIPT),
        "--candidate-run",
        str(outdir),
        "--candidate-readout",
        str(outdir / "classifier_only_readout"),
    ]


def build_pool_audit_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    return [
        python_exe,
        str(POOL_AUDIT_SCRIPT),
        "--metadata",
        str(resolve(config["paths"]["metadata_path"])),
    ]


def build_schedule_phase_command(config_path: Path, config: Dict[str, Any], python_exe: str) -> List[str]:
    outdir = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(SCHEDULE_PHASE_SCRIPT),
        "--config",
        str(config_path),
        "--run-dir",
        str(outdir),
        "--readout-dir",
        str(outdir / "classifier_only_readout"),
    ]


def validate_stage_a_command(command: Sequence[str]) -> None:
    for idx, token in enumerate(command[:-1]):
        if token.startswith("--n_iter_") and str(command[idx + 1]) == "0":
            raise RuntimeError(f"Invalid Stage A command contains {token} 0")
    require_equal(
        values_after_flag(command, "--channels_to_use"), ["1", "0", "2"], "Stage A channels"
    )
    require_equal(
        values_after_flag(command, "--classifier_types"), ["logreg", "svm"], "Stage A classifier_types"
    )
    require_equal(values_after_flag(command, "--n_iter_logreg"), ["300"], "Stage A n_iter_logreg")
    require_equal(values_after_flag(command, "--n_iter_svm"), ["300"], "Stage A n_iter_svm")
    for flag, expected in [
        ("--outer_folds", ["5"]),
        ("--inner_folds", ["5"]),
        ("--latent_dim", ["256"]),
        ("--epochs_vae", [str(EXPECTED_EPOCHS_VAE)]),
        ("--cyclical_beta_n_cycles", [str(EXPECTED_N_CYCLES)]),
        ("--lr_scheduler_T0", [str(EXPECTED_T0)]),
        ("--beta_vae", ["2.5"]),
        ("--cyclical_beta_ratio_increase", ["0.4"]),
        ("--dropout_rate_vae", ["0.15"]),
        ("--vae_dropout_scope", ["legacy_all"]),
        ("--vae_block_order", ["legacy_act_norm"]),
        ("--recon_loss_mode", ["mse_sum_batchmean_current"]),
        ("--vae_final_activation", ["tanh"]),
        ("--intermediate_fc_dim_vae", ["quarter"]),
        ("--metadata_features", ["Age", "Sex"]),
        ("--classifier_stratify_cols", ["Manufacturer"]),
        ("--vae_stratify_cols", ["Manufacturer"]),
    ]:
        require_equal(values_after_flag(command, flag), expected, f"Stage A {flag}")
    if "--vae_train_sampler_strategy" in command:
        idx = command.index("--vae_train_sampler_strategy")
        sampler_val = command[idx + 1] if idx + 1 < len(command) else ""
        if sampler_val != "none":
            raise RuntimeError(
                f"vae_train_sampler_strategy must be 'none', got {sampler_val!r}"
            )
    meta_path_values = values_after_flag(command, "--metadata_path")
    if not meta_path_values:
        raise RuntimeError("Stage A command missing --metadata_path")
    meta_path_str = meta_path_values[0]
    if not meta_path_str.endswith("patched_metadata_candidate.csv"):
        raise RuntimeError(
            f"Stage A --metadata_path must end with patched_metadata_candidate.csv, "
            f"got: {meta_path_str!r}"
        )
    require_equal(
        values_after_flag(command, "--vae_required_metadata_cols"),
        REQUIRED_METADATA_COLS,
        "Stage A --vae_required_metadata_cols",
    )
    if "--vae_abort_if_val_split_fails" not in command:
        raise RuntimeError(
            "Stage A command missing --vae_abort_if_val_split_fails. "
            "This flag is required to abort hard if the VAE val split fails "
            "(128_S_2002 is in tensor but absent from patched metadata)."
        )


def validate_stage_b_command(command: Sequence[str]) -> None:
    require_equal(values_after_flag(command, "--outer-folds"), ["5"], "Stage B outer-folds")
    require_equal(values_after_flag(command, "--inner-folds"), ["5"], "Stage B inner-folds")
    require_equal(
        values_after_flag(command, "--models"), [EXPECTED_PRIMARY_MODEL], "Stage B models"
    )


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
        raise RuntimeError(
            f"Refusing to start training: output_dir is not a symlink: {output_dir}"
        )
    if output_dir.resolve() != big_disk.resolve():
        raise RuntimeError(
            f"Refusing to start training: symlink target is {output_dir.resolve()}, "
            f"expected {big_disk.resolve()}"
        )
    stale = stale_output_markers(output_dir)
    if stale and not force_clean:
        preview = "\n".join(f"  - {p}" for p in stale[:20])
        more = "" if len(stale) <= 20 else f"\n  ... {len(stale) - 20} more"
        raise RuntimeError(
            "Refusing to start training: output_dir contains stale run artifacts. "
            "Pass --force-clean to quarantine existing contents before training.\n"
            f"{preview}{more}"
        )
    quarantine: Path | None = None
    if force_clean:
        existing = list(output_dir.iterdir())
        if existing:
            quarantine = quarantine_existing_output_contents(output_dir)
            print(f"[clean-run] Existing output_dir contents moved to quarantine: {quarantine}")
    unexpected = [
        p.name
        for p in output_dir.iterdir()
        if p.name not in {"run_manifest.json", "command_log.json"}
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
        rows.append(
            {
                "fold": fold,
                "path": str(ckpt),
                "exists": bool(exists),
                "mtime_epoch": mtime,
                "mtime_iso": datetime.fromtimestamp(mtime).isoformat() if mtime is not None else "",
                "fresh_after_run_start": bool(
                    exists and mtime is not None and mtime > run_start_epoch
                ),
            }
        )
    stale = [row for row in rows if not row["fresh_after_run_start"]]
    validation_path = output_dir / "fresh_checkpoint_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "run_start_epoch": run_start_epoch,
                "run_start_iso": datetime.fromtimestamp(run_start_epoch).isoformat(),
                "rows": rows,
                "all_fresh": not stale,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if stale:
        details = "\n".join(
            f"  - fold {row['fold']}: exists={row['exists']} mtime={row['mtime_iso']} path={row['path']}"
            for row in stale
        )
        raise RuntimeError(
            "Refusing to run Stage B: not all VAE checkpoints were created after this launcher invocation.\n"
            f"{details}"
        )
    return rows


def write_manifest(
    config_path: Path,
    config: Dict[str, Any],
    stage_a: Sequence[str],
    stage_b: Sequence[str],
    comparison: Sequence[str],
    integrity_audit: Sequence[str],
    schedule_phase: Sequence[str],
    quarantine_dir: Path | None = None,
) -> Path:
    output_dir = resolve(config["paths"]["output_dir"])
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "run_name": config["run_name"],
        "controlled_change": (
            "metadata_path -> patched_metadata_candidate.csv (adds 035_S_6927 AD); "
            "epochs_vae 4480->4500; cyclical_beta_n_cycles 56->50; lr_scheduler_T0 80->90; "
            "synchronized beta/LR cycle length = 90 epochs; "
            "pool guards: --vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex "
            "--vae_abort_if_val_split_fails"
        ),
        "only_scheduler_parameter_diff_verified": True,
        "metadata_path_changed_to_patched": True,
        "cycle_length_beta": EXPECTED_CYCLE_LEN,
        "cycle_length_lr_T0": EXPECTED_T0,
        "cycle_length_synchronized": True,
        "channels_to_use": EXPECTED_CHANNELS,
        "selected_channel_names": EXPECTED_SELECTED_NAMES,
        "python_bandpass_applied": False,
        "recover_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDED_SUBJECT,
        "expected_vae_pool": EXPECTED_VAE_N,
        "expected_clf_pool": EXPECTED_CLF_N,
        "promotion_rule": (
            f"Must beat BOTH AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} simultaneously. "
            "Reference: locked v5.1b horizon4480/cycles56 FULL [1,0,2]. FAST result is NOT sufficient."
        ),
        "stage_a_command": list(stage_a),
        "stage_a_command_shell": shlex.join(stage_a),
        "stage_b_command": list(stage_b),
        "stage_b_command_shell": shlex.join(stage_b),
        "comparison_command": list(comparison),
        "comparison_command_shell": shlex.join(comparison),
        "integrity_audit_command": list(integrity_audit),
        "integrity_audit_command_shell": shlex.join(integrity_audit),
        "schedule_phase_command": list(schedule_phase),
        "schedule_phase_command_shell": shlex.join(schedule_phase),
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else "",
        "ranking_source": "Stage B classifier-only logreg_l2",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
    }
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_command_log(
    config: Dict[str, Any],
    stage_a_rc: int | None,
    stage_b_rc: int | None,
    comparison_rc: int | None,
    quarantine_dir: Path | None = None,
) -> None:
    output_dir = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": config["run_name"],
        "training_launched": stage_a_rc is not None,
        "stage_a_returncode": stage_a_rc,
        "stage_b_returncode": stage_b_rc,
        "comparison_returncode": comparison_rc,
        "controlled_change": (
            "metadata_path -> patched_metadata_candidate.csv (adds 035_S_6927); "
            "epochs_vae 4480->4500; cyclical_beta_n_cycles 56->50; lr_scheduler_T0 80->90; "
            "synchronized 90-epoch beta/LR cycles; pool guards applied"
        ),
        "recover_subject": RECOVER_SUBJECT,
        "excluded_subject": EXCLUDED_SUBJECT,
        "promotion_rule": (
            f"Must beat BOTH AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} simultaneously."
        ),
        "ranking_source": "Stage B classifier-only logreg_l2",
        "primary_threshold_strategy": EXPECTED_PRIMARY_THRESHOLD,
        "threshold_selection": "true_inner_cv_oof",
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else "",
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
    }
    (output_dir / "command_log.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _print_schedule_phase_analysis(params: Dict[str, Any]) -> None:
    epochs = int(params["epochs_vae"])
    n_cycles = int(params["cyclical_beta_n_cycles"])
    T0 = int(params["lr_scheduler_T0"])
    cycle_len = epochs / n_cycles
    ratio_inc = float(params["cyclical_beta_ratio_increase"])
    ramp_epochs = cycle_len * ratio_inc
    stable_epochs = cycle_len * (1.0 - ratio_inc)
    remainder = epochs % int(cycle_len) if cycle_len == int(cycle_len) else float("nan")
    cycle_sync = cycle_len == T0
    restart_epochs = [i * int(cycle_len) for i in range(min(7, n_cycles + 1))]
    print("\nSchedule-phase analysis (config-only, pre-training):")
    print(
        f"  beta cycle length    = {epochs} / {n_cycles} = {cycle_len:.1f} epochs  "
        f"[{'PASS' if cycle_len == EXPECTED_CYCLE_LEN else f'FAIL: expected {EXPECTED_CYCLE_LEN}'}]"
    )
    print(
        f"  lr_scheduler_T0      = {T0} epochs  "
        f"[{'PASS' if T0 == EXPECTED_T0 else f'FAIL: expected {EXPECTED_T0}'}]"
    )
    print(
        f"  beta/LR synchronized : {cycle_sync}  "
        f"[{'PASS (both 90 epochs)' if cycle_sync else 'FAIL: not synchronized'}]"
    )
    print(f"  phase at epoch 0     : beta_phase=0.0, LR_phase=0.0  [SYNCHRONIZED]")
    print(
        f"  complete cycles      : {n_cycles}  (epochs 0 to {n_cycles * int(cycle_len) - 1})"
    )
    print(
        f"  remainder            : {int(remainder) if remainder == remainder else 'n/a'} epochs  "
        f"[{'PASS (none)' if remainder == 0 else 'WARNING: non-zero remainder'}]"
    )
    print(
        f"  per-cycle structure  : {ramp_epochs:.0f} ramp + {stable_epochs:.0f} stable = "
        f"{cycle_len:.0f} epochs (ratio_increase={ratio_inc})"
    )
    print(
        f"  restart epochs       : {', '.join(str(e) for e in restart_epochs)}"
        f"{'...' if n_cycles > 6 else ''}"
    )
    print(f"  locked reference     : 80-epoch cycles (4480/56=80, T0=80)")
    print(f"  candidate change     : 90-epoch cycles (+12.5% longer cycles, -1.1% total epochs)")


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
    if not preview_summary["passes_required_representation_check"].all():
        bad = preview_summary[~preview_summary["passes_required_representation_check"]]
        raise RuntimeError(
            "Split preview failed representation checks:\n" + bad.to_string(index=False)
        )

    recover_in_test = preview_subjects[
        (preview_subjects["SubjectID"] == RECOVER_SUBJECT)
        & (preview_subjects["split_component"] == "classifier_test")
    ]
    recover_fold = (
        int(recover_in_test["fold"].iloc[0]) if not recover_in_test.empty else None
    )

    if not args.skip_preview_write:
        preview_csv = resolve(config["paths"]["split_preview_csv"])
        preview_summary_csv = resolve(config["paths"]["split_preview_summary_csv"])
        preview_csv.parent.mkdir(parents=True, exist_ok=True)
        preview_subjects.to_csv(preview_csv, index=False)
        preview_summary.to_csv(preview_summary_csv, index=False)

    stage_a = build_stage_a_command(config, python_exe)
    stage_b = build_stage_b_command(config, python_exe)
    comparison = build_comparison_command(config, python_exe)
    integrity_audit = build_integrity_audit_command(config, python_exe)
    pool_audit = build_pool_audit_command(config, python_exe)
    schedule_phase = build_schedule_phase_command(args.config, config, python_exe)
    params = config["parameters"]
    output_dir = resolve(config["paths"]["output_dir"])
    stale = stale_output_markers(output_dir)
    mode = "DRY-RUN" if args.dry_run else "REAL RUN (CONFIRMED)"

    print(f"Config         : {args.config}")
    print(f"Source config  : {args.source_config}")
    print(f"Run name       : {config['run_name']}")
    print(f"Mode           : {mode}")
    print("Controlled diff: metadata_path -> patched_metadata_candidate.csv (035_S_6927 rescued);")
    print("                 epochs_vae 4480->4500; cyclical_beta_n_cycles 56->50; lr_scheduler_T0 80->90")
    print(f"Metadata       : {resolve(config['paths']['metadata_path'])}")
    print(
        f"Recovered subj : {RECOVER_SUBJECT} (AD, Age={EXPECTED_RESCUE_AGE}, "
        f"Sex={EXPECTED_RESCUE_SEX}, Manufacturer={EXPECTED_RESCUE_MANUFACTURER}, "
        f"tensor_idx={EXPECTED_RESCUE_TENSOR_IDX})"
    )
    print(f"Excluded subj  : {EXCLUDED_SUBJECT} (remains excluded; absent from patched metadata)")
    print(
        f"Channels       : {EXPECTED_CHANNELS} ({', '.join(EXPECTED_SELECTED_NAMES)})"
    )
    print("Recon loss     : mse_sum_batchmean_current (unchanged)")
    print(f"Block order    : {params['vae_block_order']} (unchanged)")
    print("Python bandpass: OFF")
    print("Split strategy : ResearchGroup_Mapped + Manufacturer")
    print("Sex role       : metadata/covariate only")
    print(
        f"VAE schedule   : outer={params['outer_folds']}, inner={params['inner_folds']}, "
        f"latent_dim={params['latent_dim']}, epochs={params['epochs_vae']}, "
        f"cycles={params['cyclical_beta_n_cycles']}, "
        f"cycle_len={params['epochs_vae'] / params['cyclical_beta_n_cycles']:.0f}, "
        f"T0={params['lr_scheduler_T0']}, batch_size={params['batch_size']}, "
        f"dropout={params['dropout_rate_vae']}, beta_max={params['beta_vae']}"
    )
    print(
        f"Stage A clf    : canonical {params['classifier_types']}, "
        f"n_iter_logreg={params['n_iter_logreg']}, n_iter_svm={params['n_iter_svm']}"
    )
    print(f"Stage B readout: {EXPECTED_PRIMARY_MODEL} + true inner-CV OOF {EXPECTED_PRIMARY_THRESHOLD}")
    print(
        f"Tensor shape   : {tensor_info['shape']}, n_subjects={tensor_info['n_subjects']}, "
        f"python_bandpass_applied={tensor_info['python_bandpass_applied']}"
    )
    print(
        f"VAE pool       : n={EXPECTED_VAE_N} (CN={EXPECTED_VAE_CN}, "
        f"MCI={EXPECTED_VAE_MCI}, AD={EXPECTED_VAE_AD})"
    )
    print(f"CLF pool       : n={EXPECTED_CLF_N} (CN={EXPECTED_CLF_CN}, AD={EXPECTED_CLF_AD})")
    print(f"035_S_6927 outer-test fold: {recover_fold}")
    print(
        f"Promotion rule : AUC > {LOCKED_AUC} AND PR-AUC >= {LOCKED_PR_AUC} simultaneously "
        "(FULL only; exploratory; FAST not sufficient)"
    )

    print("\nPOOL + METADATA GUARDS:")
    print(f"  Recovered subject  : {RECOVER_SUBJECT} (AD, tensor_index=256) — present in patched metadata")
    print(f"  Excluded subject   : {EXCLUDED_SUBJECT} (tensor_index=363) — absent from patched metadata")
    print(f"  Tensor n_subjects  : {tensor_info['n_subjects']} (includes {EXCLUDED_SUBJECT} with no metadata row)")
    print(
        f"  Stage A injection  : --vae_required_metadata_cols {' '.join(REQUIRED_METADATA_COLS)}"
    )
    print(
        "                       --vae_abort_if_val_split_fails"
    )
    print("  Effect: 128_S_2002 NaN metadata row filtered from VAE pool before split attempt")
    print()
    print("[Expected per-fold VAE pool sizes after --vae_required_metadata_cols filtering]")
    print("  (Patched metadata n=647 minus test CN/AD per fold; 128_S_2002 filtered from tensor join)")
    vae_pool_rows = preview_summary[preview_summary["split_component"] == "vae_pool"]
    vae_train_rows = preview_summary[preview_summary["split_component"] == "vae_actual_train"]
    vae_val_rows = preview_summary[preview_summary["split_component"] == "vae_internal_val"]
    for fold_num in sorted(preview_summary["fold"].unique()):
        pool_n = (
            int(vae_pool_rows[vae_pool_rows["fold"] == fold_num]["n"].iloc[0])
            if not vae_pool_rows[vae_pool_rows["fold"] == fold_num].empty
            else "?"
        )
        train_n = (
            int(vae_train_rows[vae_train_rows["fold"] == fold_num]["n"].iloc[0])
            if not vae_train_rows[vae_train_rows["fold"] == fold_num].empty
            else "?"
        )
        val_n = (
            int(vae_val_rows[vae_val_rows["fold"] == fold_num]["n"].iloc[0])
            if not vae_val_rows[vae_val_rows["fold"] == fold_num].empty
            else "?"
        )
        print(f"  Fold {fold_num}: VAE pool={pool_n}, actual_train={train_n}, internal_val={val_n}")

    _print_schedule_phase_analysis(params)

    print("\n[split] All folds/components have required diagnosis/manufacturer representation.")
    preview_cols = [
        "fold", "split_component", "n", "AD", "CN", "MCI",
        "Manufacturer_GE", "Manufacturer_Siemens", "Manufacturer_Philips",
        "Sex_F", "Sex_M",
        f"contains_{RECOVER_SUBJECT.replace('-', '_')}",
    ]
    print(
        preview_summary[[c for c in preview_cols if c in preview_summary.columns]].to_string(
            index=False
        )
    )

    print("\nStage A command:")
    print(shlex.join(stage_a))
    print("\nStage B classifier-only readout command:")
    print(shlex.join(stage_b))
    print("\nComparison command:")
    print(shlex.join(comparison))
    print("\nIntegrity audit command:")
    print(shlex.join(integrity_audit))
    print("\nPool audit command:")
    print(shlex.join(pool_audit))
    print("\nSchedule-phase audit command:")
    print(shlex.join(schedule_phase))
    print("\nClean-run policy:")
    print(
        "Real training refuses any existing fold_*, classifier_only_readout, latent_cache, "
        "all_folds_metrics, summary_metrics, or run_manifest.json unless --force-clean is passed."
    )
    print("\nRESTART INSTRUCTIONS (if partial artifacts exist from a failed run):")
    print("  1. Verify the partial run is invalid (empty val set, or 128_S_2002 in VAE pool).")
    print(f"  2. python {Path(__file__).name} --dry-run   # confirm stale markers detected")
    print(
        f"  3. python {Path(__file__).name} --force-clean --confirm-training   "
        "# quarantine + rerun cleanly"
    )
    if stale:
        print(f"Detected stale markers in output_dir: {len(stale)}")
        for marker in stale[:20]:
            print(f"  - {marker}")
        if len(stale) > 20:
            print(f"  ... {len(stale) - 20} more")
        if args.dry_run:
            print("Dry-run only: no quarantine or cleanup was performed.")
    else:
        print("No stale markers detected in output_dir.")

    if args.dry_run:
        print("\nDry-run complete. Training was NOT launched.")
        return 0

    quarantine_dir = ensure_output_prepared(config, force_clean=args.force_clean)
    run_start_epoch = time.time()
    manifest = write_manifest(
        args.config,
        config,
        stage_a,
        stage_b,
        comparison,
        integrity_audit,
        schedule_phase,
        quarantine_dir=quarantine_dir,
    )
    print(f"\nRun manifest written: {manifest}")
    completed = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False)
    write_command_log(config, completed.returncode, None, None, quarantine_dir=quarantine_dir)
    if completed.returncode != 0:
        return int(completed.returncode)
    fresh_rows = verify_fresh_vae_checkpoints(config, run_start_epoch)
    print(
        "[fresh-checkpoint] All fold_1..fold_5 VAE checkpoints exist and are newer than this run start."
    )
    for row in fresh_rows:
        print(f"  - fold {row['fold']}: {row['mtime_iso']} {row['path']}")
    if args.skip_classifier_readout:
        print("Stage A completed. Stage B skipped.")
        return 0
    readout_completed = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False)
    write_command_log(
        config, completed.returncode, readout_completed.returncode, None, quarantine_dir=quarantine_dir
    )
    if readout_completed.returncode != 0:
        return int(readout_completed.returncode)
    print("\nStage B completed. Run these audits before considering promotion:")
    print("  integrity audit :", shlex.join(integrity_audit))
    print("  schedule phase  :", shlex.join(schedule_phase))
    if args.skip_comparison:
        print("Stage B completed. Comparison skipped.")
        return 0
    comparison_completed = subprocess.run(comparison, cwd=PROJECT_ROOT, check=False)
    write_command_log(
        config,
        completed.returncode,
        readout_completed.returncode,
        comparison_completed.returncode,
        quarantine_dir=quarantine_dir,
    )
    return int(comparison_completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
