#!/usr/bin/env python3
"""Prepare/run the controlled Manufacturer-balanced VAE sampler sensitivity.

New run:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE

Reference promoted run:
  recover035_latent384_beta3p75_T80_h10000_p560_full5x5

Scientific change:
  vae_train_sampler_strategy: none -> manufacturer_balanced

The sampler is fit only on each fold's VAE actual-training rows, uses
Manufacturer only as the balancing key, and never uses diagnosis labels in the
VAE loss or sampler key.

Default behavior is preflight only. Real training requires --confirm-training.
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
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betavae_xai.models import (  # noqa: E402
    ConvolutionalVAE,
    build_vae_dropout_manifest,
    summarize_vae_dropout_manifest,
)


REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE.json"
)
DEFAULT_PREFLIGHT_DIR = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE_preflight"
)

RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_METADATA_FEATURES = ["Age", "Sex"]
EXPECTED_STRATIFY_COLS = ["Manufacturer"]
EXPECTED_CYCLE_LEN = 80
EXPECTED_PATIENCE_CYCLES = 7
EXPECTED_SAMPLER_STRATEGY = "manufacturer_balanced"
EXPECTED_DROPOUT_SUMMARY = {
    "encoder_conv": 4,
    "encoder_fc": 1,
    "decoder_fc": 1,
    "decoder_conv": 3,
}
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934

STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
OOF_SCORE_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    parser.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real training.")
    parser.add_argument("--force-clean", action="store_true", help="Quarantine existing output contents before real training.")
    parser.add_argument("--skip-stage-b", action="store_true")
    parser.add_argument("--skip-oof-score-harmonization", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


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
    for token in tokens[tokens.index(flag) + 1 :]:
        if token.startswith("--"):
            break
        out.append(token)
    return out


def normalize_mfr(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    up = text.upper()
    if "GE" in up:
        return "GE"
    if "SIEMENS" in up:
        return "SIEMENS"
    if "PHILIPS" in up:
        return "Philips"
    return text or "UNKNOWN"


def effective_config_diff(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    ref = dict(reference["parameters"])
    cand = dict(candidate["parameters"])
    for params in (ref, cand):
        params.setdefault("recon_loss_mode", "mse_sum_batchmean_current")
        params.setdefault("vae_dropout_scope", "legacy_all")
        params.setdefault("vae_block_order", "legacy_act_norm")
        params.setdefault("vae_train_sampler_strategy", "none")
    diffs = {k: (ref.get(k), cand.get(k)) for k in sorted(set(ref) | set(cand)) if ref.get(k) != cand.get(k)}
    for section in ["selected_channel_names", "channel_names_master_in_tensor_order", "split_strategy"]:
        if reference.get(section) != candidate.get(section):
            diffs[section] = (reference.get(section), candidate.get(section))
    for path_key in ["training_script", "global_tensor_path", "metadata_path"]:
        if reference["paths"].get(path_key) != candidate["paths"].get(path_key):
            diffs[f"paths.{path_key}"] = (reference["paths"].get(path_key), candidate["paths"].get(path_key))
    return diffs


def validate_config(config: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    params = config["parameters"]
    diffs = effective_config_diff(reference, config)
    require_equal(
        diffs,
        {"vae_train_sampler_strategy": ("none", EXPECTED_SAMPLER_STRATEGY)},
        "effective config diff vs promoted latent384",
    )
    require_equal(
        config["run_name"],
        "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_mfrBalancedVAE",
        "run_name",
    )
    require_equal(params["latent_dim"], 384, "latent_dim")
    require_equal(params["beta_vae"], 3.75, "beta_vae")
    require_equal(params["dropout_rate_vae"], 0.15, "dropout_rate_vae")
    require_equal(params.get("vae_dropout_scope", "legacy_all"), "legacy_all", "vae_dropout_scope")
    require_equal(params["channels_to_use"], EXPECTED_CHANNELS, "channels_to_use")
    require_equal(config["selected_channel_names"], EXPECTED_SELECTED_NAMES, "selected_channel_names")
    require_equal(params["epochs_vae"], 10000, "epochs_vae")
    require_equal(params["cyclical_beta_n_cycles"], 125, "cyclical_beta_n_cycles")
    require_equal(params["lr_scheduler_T0"], 80, "lr_scheduler_T0")
    require_equal(params["early_stopping_patience_vae"], 560, "early_stopping_patience_vae")
    require_equal(params["batch_size"], 64, "batch_size")
    require_equal(params["intermediate_fc_dim_vae"], "quarter", "intermediate_fc_dim_vae")
    require_equal(params["decoder_type"], "convtranspose", "decoder_type")
    require_equal(params["recon_loss_mode"], "mse_sum_batchmean_current", "recon_loss_mode")
    require_equal(params["norm_mode"], "zscore_offdiag", "norm_mode")
    require_equal(params["metadata_features"], EXPECTED_METADATA_FEATURES, "metadata_features")
    require_equal(params["classifier_stratify_cols"], EXPECTED_STRATIFY_COLS, "classifier_stratify_cols")
    require_equal(params["vae_stratify_cols"], EXPECTED_STRATIFY_COLS, "vae_stratify_cols")
    require_equal(
        params.get("vae_train_sampler_strategy", "none"),
        EXPECTED_SAMPLER_STRATEGY,
        "vae_train_sampler_strategy",
    )
    cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    require_equal(cycle_len, EXPECTED_CYCLE_LEN, "cycle_len")
    require_equal(params["lr_scheduler_T0"], EXPECTED_CYCLE_LEN, "T0 equals cycle_len")
    require_equal(params["early_stopping_patience_vae"] // EXPECTED_CYCLE_LEN, EXPECTED_PATIENCE_CYCLES, "patience cycles")
    return diffs


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
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Sex"] = meta["Sex"].fillna("UNKNOWN").astype(str)
    meta["tensor_idx"] = meta["tensor_idx"].astype(int)
    if meta["SubjectID"].duplicated().any():
        raise RuntimeError("Metadata has duplicate SubjectID rows.")
    return meta


def audit_subject_pool(meta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dx_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf_counts = clf["ResearchGroup_Mapped"].value_counts().to_dict()
    checks = [
        ("vae_CN", dx_counts.get("CN", 0), EXPECTED_VAE_COUNTS["CN"]),
        ("vae_MCI", dx_counts.get("MCI", 0), EXPECTED_VAE_COUNTS["MCI"]),
        ("vae_AD", dx_counts.get("AD", 0), EXPECTED_VAE_COUNTS["AD"]),
        ("classifier_CN", clf_counts.get("CN", 0), EXPECTED_CLF_COUNTS["CN"]),
        ("classifier_AD", clf_counts.get("AD", 0), EXPECTED_CLF_COUNTS["AD"]),
        ("035_present", int(meta["SubjectID"].eq(RECOVER_SUBJECT).any()), 1),
        ("128_absent", int(~meta["SubjectID"].eq(EXCLUDED_SUBJECT).any()), 1),
    ]
    for name, actual, expected in checks:
        rows.append({"check": name, "actual": actual, "expected": expected, "pass": actual == expected})
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Subject pool audit failed:\n" + out.to_string(index=False))
    return out


def inspect_tensor(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as zf:
        shape = tuple(int(x) for x in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_ids = zf["subject_ids"].astype(str).tolist()
        py_bandpass = bool(zf["python_bandpass_applied"]) if "python_bandpass_applied" in zf.files else None
    selected = [channel_names[i] for i in EXPECTED_CHANNELS]
    require_equal(selected, EXPECTED_SELECTED_NAMES, "selected tensor channels")
    if RECOVER_SUBJECT not in subject_ids:
        raise RuntimeError(f"{RECOVER_SUBJECT} missing from tensor")
    return {
        "tensor_path": str(path),
        "shape": shape,
        "n_subjects": len(subject_ids),
        "n_channels": shape[1],
        "n_roi_1": shape[2],
        "n_roi_2": shape[3],
        "selected_channel_names": selected,
        "python_bandpass_applied": py_bandpass,
        "recover035_tensor_index": subject_ids.index(RECOVER_SUBJECT),
    }


def strat_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    tmp = df[list(cols)].copy()
    for col in cols:
        tmp[col] = tmp[col].fillna(f"{col}_UNKNOWN").astype(str)
    return tmp.apply(lambda row: "_".join(row.values.astype(str)), axis=1)


def count_split(df: pd.DataFrame) -> Dict[str, Any]:
    row: Dict[str, Any] = {"n": int(len(df))}
    for dx in ["CN", "MCI", "AD"]:
        row[dx] = int(df["ResearchGroup_Mapped"].eq(dx).sum())
    for mfr in ["GE", "SIEMENS", "Philips"]:
        label = "Siemens" if mfr == "SIEMENS" else mfr
        row[f"Manufacturer_{label}"] = int(df["Manufacturer"].eq(mfr).sum())
        for dx in ["CN", "MCI", "AD"]:
            row[f"{dx}_{label}"] = int((df["ResearchGroup_Mapped"].eq(dx) & df["Manufacturer"].eq(mfr)).sum())
    row["contains_035_S_6927"] = bool(df["SubjectID"].eq(RECOVER_SUBJECT).any())
    row["contains_128_S_2002"] = bool(df["SubjectID"].eq(EXCLUDED_SUBJECT).any())
    return row


def validate_split_component(row: Dict[str, Any], component: str) -> str:
    diagnoses = ["CN", "AD"] if component in {"classifier_train_dev", "classifier_test"} else ["CN", "MCI", "AD"]
    errors: List[str] = []
    for dx in diagnoses:
        if row[dx] <= 0:
            errors.append(f"missing {dx}")
    for label in ["GE", "Siemens", "Philips"]:
        if row[f"Manufacturer_{label}"] <= 0:
            errors.append(f"missing {label}")
    if row["contains_128_S_2002"]:
        errors.append("contains excluded 128_S_2002")
    return " | ".join(errors)


def make_split_preview(meta: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    seed = int(params["seed"])
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    outer_cols = ["ResearchGroup_Mapped", *params["classifier_stratify_cols"]]
    outer_key = strat_key(cn_ad, outer_cols)
    if (outer_key.value_counts() < params["outer_folds"]).any():
        raise RuntimeError("Outer Manufacturer-stratified split infeasible.")
    by_tensor = meta.set_index("tensor_idx", drop=False)
    all_tensor_idx = meta["tensor_idx"].to_numpy()
    split_rows: List[Dict[str, Any]] = []
    subject_rows: List[Dict[str, Any]] = []
    sampler_rows: List[Dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=params["outer_folds"], shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(cn_ad)), outer_key), start=1):
        train_dev = cn_ad.iloc[train_idx].copy()
        test = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(all_tensor_idx, test["tensor_idx"].to_numpy(), assume_unique=False)
        vae_pool = by_tensor.loc[vae_pool_idx].reset_index(drop=True)
        vae_key = strat_key(vae_pool, ["ResearchGroup_Mapped", *params["vae_stratify_cols"]])
        if (vae_key.value_counts() < 2).any():
            raise RuntimeError(f"Fold {fold}: VAE internal validation split infeasible.")
        vae_train_idx, vae_val_idx = train_test_split(
            np.arange(len(vae_pool)),
            test_size=float(params["vae_val_split_ratio"]),
            stratify=vae_key,
            random_state=seed + fold + 9,
            shuffle=True,
        )
        vae_actual_train = vae_pool.iloc[vae_train_idx].copy()
        test_subjects = set(test["SubjectID"].astype(str))
        sampler_subjects = set(vae_actual_train["SubjectID"].astype(str))
        test_overlap = sorted(test_subjects & sampler_subjects)
        if test_overlap:
            raise RuntimeError(
                f"Fold {fold}: classifier outer-test subjects entered VAE sampler fit rows: {test_overlap[:8]}"
            )
        mfr_counts = vae_actual_train["Manufacturer"].fillna("UNKNOWN").astype(str).value_counts().sort_index()
        n_groups = int(len(mfr_counts))
        n_train = int(len(vae_actual_train))
        for mfr, n_mfr in mfr_counts.items():
            dx_counts = (
                vae_actual_train.loc[vae_actual_train["Manufacturer"].astype(str) == str(mfr), "ResearchGroup_Mapped"]
                .value_counts()
                .to_dict()
            )
            sampler_rows.append(
                {
                    "fold": fold,
                    "sampler_strategy": EXPECTED_SAMPLER_STRATEGY,
                    "sampler_fit_scope": "vae_actual_train_only",
                    "sampler_key_columns": "Manufacturer",
                    "uses_diagnosis_labels_in_sampler_key": False,
                    "outer_test_subject_overlap_count": len(test_overlap),
                    "Manufacturer": mfr,
                    "n_vae_actual_train": n_train,
                    "n_manufacturer": int(n_mfr),
                    "n_sampler_groups": n_groups,
                    "expected_normalized_weight_per_subject": float(n_train / (n_groups * int(n_mfr))),
                    "diagnosis_counts_in_group_descriptive_only": json.dumps(dx_counts, sort_keys=True),
                }
            )
        components = [
            ("classifier_train_dev", train_dev),
            ("classifier_test", test),
            ("vae_pool", vae_pool),
            ("vae_actual_train", vae_actual_train),
            ("vae_internal_val", vae_pool.iloc[vae_val_idx]),
        ]
        for component, df in components:
            row = {"fold": fold, "split_component": component}
            row.update(count_split(df))
            row["representation_errors"] = validate_split_component(row, component)
            row["passes_required_representation_check"] = row["representation_errors"] == ""
            split_rows.append(row)
        for split_name, df in [("classifier_train_dev", train_dev), ("classifier_test", test)]:
            for _, r in df.iterrows():
                subject_rows.append(
                    {
                        "fold": fold,
                        "split_component": split_name,
                        "SubjectID": r["SubjectID"],
                        "tensor_idx": int(r["tensor_idx"]),
                        "ResearchGroup_Mapped": r["ResearchGroup_Mapped"],
                        "Manufacturer": r["Manufacturer"],
                        "Age": r["Age"],
                        "Sex": r["Sex"],
                    }
                )
    preview = pd.DataFrame(split_rows)
    if not preview["passes_required_representation_check"].all():
        raise RuntimeError("Split preview failed:\n" + preview.to_string(index=False))
    return preview, pd.DataFrame(subject_rows), pd.DataFrame(sampler_rows)


def make_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    paths = config["paths"]
    cmd = [
        python_exe,
        str(resolve(paths["training_script"])),
        "--global_tensor_path",
        str(resolve(paths["global_tensor_path"])),
        "--metadata_path",
        str(resolve(paths["metadata_path"])),
        "--output_dir",
        str(resolve(paths["output_dir"])),
    ]
    for key, value in config["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    return cmd


def make_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    params = config["parameters"]
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir",
        str(out),
        "--output-dir",
        str(out / "classifier_only_readout"),
        "--outer-folds",
        str(params["outer_folds"]),
        "--inner-folds",
        str(params["inner_folds"]),
        "--models",
        PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]


def make_oof_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(OOF_SCORE_SCRIPT),
        "--run-dir",
        str(out),
        "--output-dir",
        str(PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_mfrBalancedVAE_stageB_oof_score_calibration"),
    ]


def validate_commands(stage_a: Sequence[str], stage_b: Sequence[str], oof: Sequence[str]) -> pd.DataFrame:
    checks = [
        ("stage_a_channels", values_after_flag(stage_a, "--channels_to_use"), ["1", "0", "2"]),
        ("stage_a_latent_dim", values_after_flag(stage_a, "--latent_dim"), ["384"]),
        ("stage_a_beta", values_after_flag(stage_a, "--beta_vae"), ["3.75"]),
        ("stage_a_vae_sampler", values_after_flag(stage_a, "--vae_train_sampler_strategy"), [EXPECTED_SAMPLER_STRATEGY]),
        ("stage_a_epochs", values_after_flag(stage_a, "--epochs_vae"), ["10000"]),
        ("stage_a_cycles", values_after_flag(stage_a, "--cyclical_beta_n_cycles"), ["125"]),
        ("stage_a_T0", values_after_flag(stage_a, "--lr_scheduler_T0"), ["80"]),
        ("stage_a_patience", values_after_flag(stage_a, "--early_stopping_patience_vae"), ["560"]),
        ("stage_a_dropout", values_after_flag(stage_a, "--dropout_rate_vae"), ["0.15"]),
        ("stage_a_dropout_scope", values_after_flag(stage_a, "--vae_dropout_scope"), ["legacy_all"]),
        ("stage_a_metadata_features", values_after_flag(stage_a, "--metadata_features"), ["Age", "Sex"]),
        ("stage_a_required_metadata", values_after_flag(stage_a, "--vae_required_metadata_cols"), ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]),
        ("stage_b_model", values_after_flag(stage_b, "--models"), [PRIMARY_MODEL]),
        ("stage_b_outer", values_after_flag(stage_b, "--outer-folds"), ["5"]),
        ("stage_b_inner", values_after_flag(stage_b, "--inner-folds"), ["5"]),
        ("oof_run_dir", values_after_flag(oof, "--run-dir"), [values_after_flag(stage_b, "--run-dir")[0]]),
    ]
    rows = []
    for check, actual, expected in checks:
        rows.append({"check": check, "actual": " ".join(actual), "expected": " ".join(expected), "pass": actual == expected})
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Command validation failed:\n" + out.to_string(index=False))
    return out


def output_symlink_status(config: Dict[str, Any]) -> Dict[str, Any]:
    local = resolve(config["paths"]["output_dir"])
    target = Path(config["paths"]["big_disk_output_dir"])
    status = {
        "local_output_dir": str(local),
        "big_disk_output_dir": str(target),
        "local_exists": local.exists() or local.is_symlink(),
        "local_is_symlink": local.is_symlink(),
        "big_disk_exists": target.exists(),
        "symlink_target": str(local.resolve()) if local.exists() or local.is_symlink() else "",
        "target_match": bool((local.exists() or local.is_symlink()) and local.is_symlink() and local.resolve() == target.resolve()),
    }
    return status


def stale_output_markers(output_dir: Path) -> List[Path]:
    if not output_dir.exists():
        return []
    markers = []
    for child in output_dir.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in output_dir.rglob("latent_cache"):
        markers.append(nested)
    return sorted(set(markers), key=lambda p: str(p))


def quarantine_contents(output_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = output_dir.resolve().parent / f"{output_dir.resolve().name}_quarantine_{timestamp}"
    suffix = 1
    while quarantine.exists():
        quarantine = output_dir.resolve().parent / f"{output_dir.resolve().name}_quarantine_{timestamp}_{suffix}"
        suffix += 1
    quarantine.mkdir(parents=True, exist_ok=False)
    for child in list(output_dir.iterdir()):
        child.rename(quarantine / child.name)
    return quarantine


def ensure_clean_real_output(config: Dict[str, Any], force_clean: bool) -> Path | None:
    status = output_symlink_status(config)
    local = Path(status["local_output_dir"])
    if not status["target_match"]:
        raise RuntimeError(f"Output symlink validation failed: {json.dumps(status, indent=2)}")
    stale = stale_output_markers(local)
    if stale and not force_clean:
        raise RuntimeError("Refusing real training due to stale output markers:\n" + "\n".join(str(p) for p in stale[:20]))
    quarantine = None
    if force_clean and list(local.iterdir()):
        quarantine = quarantine_contents(local)
    if list(local.iterdir()):
        raise RuntimeError("Refusing real training: output directory is not empty after clean policy.")
    return quarantine


def dropout_manifest(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    params = config["parameters"]
    model = ConvolutionalVAE(
        input_channels=len(params["channels_to_use"]),
        latent_dim=int(params["latent_dim"]),
        image_size=131,
        final_activation=params.get("vae_final_activation", "tanh"),
        intermediate_fc_dim_config=params.get("intermediate_fc_dim_vae", "quarter"),
        dropout_rate=float(params.get("dropout_rate_vae", 0.15)),
        encoder_dropout_rate=params.get("encoder_dropout_rate_vae"),
        decoder_dropout_rate=params.get("decoder_dropout_rate_vae"),
        use_layernorm_fc=bool(params.get("use_layernorm_vae_fc", False)),
        num_conv_layers_encoder=int(params.get("num_conv_layers_encoder", 4)),
        decoder_type=params.get("decoder_type", "convtranspose"),
        encoder_norm_mode=params.get("encoder_norm_mode", "groupnorm"),
        dropout_scope=params.get("vae_dropout_scope", "legacy_all"),
        block_order=params.get("vae_block_order", "legacy_act_norm"),
    )
    manifest = pd.DataFrame(build_vae_dropout_manifest(model))
    summary = pd.DataFrame(
        summarize_vae_dropout_manifest(
            manifest.to_dict("records"),
            dropout_scope=params.get("vae_dropout_scope", "legacy_all"),
            dropout_rate=float(params.get("dropout_rate_vae", 0.15)),
            num_conv_layers_encoder=int(params.get("num_conv_layers_encoder", 4)),
            has_intermediate_fc=params.get("intermediate_fc_dim_vae", "quarter") not in {"0", 0, None},
            encoder_dropout_rate=params.get("encoder_dropout_rate_vae"),
            decoder_dropout_rate=params.get("decoder_dropout_rate_vae"),
        )
    )
    observed = dict(zip(summary["location"], summary["observed_count"]))
    for loc, expected in EXPECTED_DROPOUT_SUMMARY.items():
        if int(observed.get(loc, -1)) != expected:
            raise RuntimeError(f"Dropout manifest mismatch for {loc}: expected {expected}, got {observed.get(loc)}")
    if len(manifest) != sum(EXPECTED_DROPOUT_SUMMARY.values()):
        raise RuntimeError(f"Expected 9 dropout modules, got {len(manifest)}")
    if not np.allclose(manifest["p"].astype(float), 0.15):
        raise RuntimeError("Expected all dropout p=0.15")
    return manifest, summary


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    path.with_suffix(".md").write_text(md + "\n", encoding="utf-8")


def write_preflight_outputs(
    out_dir: Path,
    config_path: Path,
    reference_config_path: Path,
    config: Dict[str, Any],
    diffs: Dict[str, Tuple[Any, Any]],
    tensor_info: Dict[str, Any],
    pool_audit: pd.DataFrame,
    split_summary: pd.DataFrame,
    split_subjects: pd.DataFrame,
    sampler_audit: pd.DataFrame,
    command_validation: pd.DataFrame,
    symlink_status: Dict[str, Any],
    stale: List[Path],
    drop_manifest: pd.DataFrame,
    drop_summary: pd.DataFrame,
    stage_a: Sequence[str],
    stage_b: Sequence[str],
    oof: Sequence[str],
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(pool_audit, out_dir / "subject_pool_audit.csv")
    write_table(split_summary, out_dir / "split_preview_summary.csv")
    count_cols = [
        "fold",
        "split_component",
        "n",
        "CN",
        "MCI",
        "AD",
        "Manufacturer_GE",
        "Manufacturer_Philips",
        "Manufacturer_Siemens",
        "passes_required_representation_check",
    ]
    write_table(split_summary[count_cols], out_dir / "fold_manufacturer_counts.csv")
    write_table(
        split_summary.loc[
            split_summary["split_component"].isin(["vae_pool", "vae_actual_train", "vae_internal_val"]),
            count_cols,
        ],
        out_dir / "vae_pool_counts_by_fold.csv",
    )
    split_subjects.to_csv(out_dir / "split_preview_subjects.csv", index=False)
    write_table(sampler_audit, out_dir / "vae_manufacturer_balanced_sampler_preview.csv")
    write_table(command_validation, out_dir / "command_validation.csv")
    write_table(drop_manifest, out_dir / "dropout_manifest_expected.csv")
    write_table(drop_summary, out_dir / "dropout_summary_expected.csv")
    diff_rows = [{"field": k, "reference": v[0], "candidate": v[1]} for k, v in diffs.items()]
    write_table(pd.DataFrame(diff_rows), out_dir / "config_diff_vs_promoted_latent384.csv")
    symlink_df = pd.DataFrame([{**symlink_status, "stale_marker_count": len(stale), "stale_markers": " | ".join(str(p) for p in stale)}])
    write_table(symlink_df, out_dir / "output_symlink_and_stale_audit.csv")
    tensor_df = pd.DataFrame([{**tensor_info}])
    write_table(tensor_df, out_dir / "tensor_compatibility_audit.csv")
    commands = pd.DataFrame(
        [
            {"name": "stage_a_training", "command": shlex.join(stage_a)},
            {"name": "stage_b_classifier_only", "command": shlex.join(stage_b)},
            {"name": "stage_b_oof_score_harmonization", "command": shlex.join(oof)},
        ]
    )
    write_table(commands, out_dir / "planned_commands.csv")
    (out_dir / "sampler_leakage_guardrail.md").write_text(
        "\n".join(
            [
                "# VAE Sampler Leakage Guardrail",
                "",
                "- Sampler strategy: `manufacturer_balanced`.",
                "- Sampler fit scope: fold-local `vae_actual_train_only` rows.",
                "- Sampler key columns: `Manufacturer` only.",
                "- Diagnosis labels are not used in the sampler key or VAE loss.",
                "- The classifier outer-test subjects are removed before VAE pool splitting and before sampler fitting.",
                "- The preflight table `vae_manufacturer_balanced_sampler_preview.csv` must show `outer_test_subject_overlap_count=0` in every row.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = [
        "# Manufacturer-Balanced VAE Sampler FULL 5x5 Preflight",
        "",
        f"Config: `{config_path}`",
        f"Reference config: `{reference_config_path}`",
        "",
        "## Decision",
        "",
        "Prepared only. Real training was not launched.",
        "",
        "## Controlled Scientific Diff",
        "",
        f"- `vae_train_sampler_strategy: none -> {EXPECTED_SAMPLER_STRATEGY}`",
        "- The VAE remains diagnosis-agnostic; diagnosis labels are not used in the VAE loss or sampler key.",
        "- `latent_dim=384`, `beta_vae=3.75`, channels, architecture, dropout, scheduler, folds, classifiers, calibration, and metadata features remain unchanged.",
        "",
        "## Validation",
        "",
        f"- JSON syntax: OK.",
        f"- Config diff: exactly `{diffs}`.",
        f"- Cycle length: `{config['parameters']['epochs_vae'] / config['parameters']['cyclical_beta_n_cycles']:.0f}` epochs.",
        f"- T0: `{config['parameters']['lr_scheduler_T0']}`.",
        f"- Patience: `{config['parameters']['early_stopping_patience_vae']}` = 7 cycles.",
        f"- Subject pool: VAE CN=300/MCI=250/AD=97; classifier CN=300/AD=97.",
        "- Manufacturer-balanced sampler preview: written to `vae_manufacturer_balanced_sampler_preview.csv`.",
        "- Sampler/test overlap: 0 by construction and checked per fold.",
        f"- 035_S_6927 included; 128_S_2002 excluded from metadata-driven VAE/classifier pools.",
        f"- Dropout manifest: 9 modules, all p=0.15, legacy_all hidden dropout.",
        f"- Output symlink valid: `{symlink_status['target_match']}`.",
        f"- Stale markers: `{len(stale)}`.",
        "",
        "## Guardrail",
        "",
        "The launcher refuses real training without `--confirm-training`; it also refuses stale output folders unless `--force-clean` is used to quarantine existing contents. The VAE sampler is fit after the classifier outer-test fold has been removed.",
    ]
    (out_dir / "preflight_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    command_log = {
        "created_utc": now_utc(),
        "dry_run": dry_run,
        "training_launched": False,
        "config_path": str(config_path),
        "reference_config_path": str(reference_config_path),
        "run_name": config["run_name"],
        "controlled_diff": {"vae_train_sampler_strategy": ["none", EXPECTED_SAMPLER_STRATEGY]},
        "stage_a_command": list(stage_a),
        "stage_b_command": list(stage_b),
        "oof_score_harmonization_command": list(oof),
        "no_tensor_modification": True,
        "no_metadata_modification": True,
        "no_training_launched": True,
    }
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_run_manifest(config: Dict[str, Any], stage_a: Sequence[str], stage_b: Sequence[str], oof: Sequence[str], quarantine: Path | None) -> None:
    out = resolve(config["paths"]["output_dir"])
    payload = {
        "created_utc": now_utc(),
        "run_name": config["run_name"],
        "controlled_change": "vae_train_sampler_strategy none -> manufacturer_balanced only versus promoted latent384 beta3p75 reference",
        "vae_sampler_fit_scope": "fold-local vae_actual_train_only",
        "vae_sampler_key_columns": ["Manufacturer"],
        "vae_sampler_uses_diagnosis_labels": False,
        "stage_a_command": list(stage_a),
        "stage_b_command": list(stage_b),
        "oof_score_harmonization_command": list(oof),
        "quarantine_dir": str(quarantine) if quarantine else "",
        "promotion_gate": "AUC > 0.795155 and PR-AUC >= 0.573934, with BA/F1/Sens and Manufacturer/Philips CN FP behavior not materially worse",
    }
    (out / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight only.")
    config = load_json(args.config)
    reference = load_json(args.reference_config)
    diffs = validate_config(config, reference)
    python_exe = args.python_executable or config.get("python_executable") or sys.executable
    tensor_info = inspect_tensor(resolve(config["paths"]["global_tensor_path"]))
    meta = load_metadata(resolve(config["paths"]["metadata_path"]))
    pool = audit_subject_pool(meta)
    split_summary, split_subjects, sampler_audit = make_split_preview(meta, config)
    split_csv = resolve(config["paths"]["split_preview_csv"])
    split_summary_csv = resolve(config["paths"]["split_preview_summary_csv"])
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split_subjects.to_csv(split_csv, index=False)
    split_summary.to_csv(split_summary_csv, index=False)
    drop_manifest, drop_summary = dropout_manifest(config)
    stage_a = make_stage_a_command(config, python_exe)
    stage_b = make_stage_b_command(config, python_exe)
    oof = make_oof_command(config, python_exe)
    command_validation = validate_commands(stage_a, stage_b, oof)
    symlink = output_symlink_status(config)
    stale = stale_output_markers(resolve(config["paths"]["output_dir"]))
    write_preflight_outputs(
        args.preflight_dir,
        args.config,
        args.reference_config,
        config,
        diffs,
        tensor_info,
        pool,
        split_summary,
        split_subjects,
        sampler_audit,
        command_validation,
        symlink,
        stale,
        drop_manifest,
        drop_summary,
        stage_a,
        stage_b,
        oof,
        dry_run=args.dry_run or not args.confirm_training,
    )
    print(f"Preflight outputs: {args.preflight_dir}")
    print(f"Controlled diff: vae_train_sampler_strategy none -> {EXPECTED_SAMPLER_STRATEGY} only.")
    print(f"Output symlink target valid: {symlink['target_match']}")
    print(f"Stale markers: {len(stale)}")
    print("Stage A command:")
    print(shlex.join(stage_a))
    print("Stage B command:")
    print(shlex.join(stage_b))
    print("OOF score-harmonization command:")
    print(shlex.join(oof))
    if args.dry_run:
        print("Dry-run complete. Training was NOT launched.")
        return 0
    quarantine = ensure_clean_real_output(config, args.force_clean)
    write_run_manifest(config, stage_a, stage_b, oof, quarantine)
    run_start = time.time()
    rc_a = subprocess.run(stage_a, cwd=PROJECT_ROOT, check=False).returncode
    if rc_a != 0:
        return rc_a
    for fold in range(1, 6):
        ckpt = resolve(config["paths"]["output_dir"]) / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists() or ckpt.stat().st_mtime <= run_start:
            raise RuntimeError(f"Fold {fold} checkpoint missing or stale after Stage A: {ckpt}")
    if not args.skip_stage_b:
        rc_b = subprocess.run(stage_b, cwd=PROJECT_ROOT, check=False).returncode
        if rc_b != 0:
            return rc_b
    if not args.skip_oof_score_harmonization:
        rc_o = subprocess.run(oof, cwd=PROJECT_ROOT, check=False).returncode
        if rc_o != 0:
            return rc_o
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
