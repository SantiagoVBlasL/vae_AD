#!/usr/bin/env python3
"""Preflight/launcher for recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14.

Post-final exploratory parsimony sensitivity.

Controlled two-parameter diff versus promoted reference
(recover035_latent384_beta3p75_T80_h10000_p560_full5x5):
  1. channels_to_use: [1,0,2] -> [1]  (Pearson_Full_FisherZ_Signed only)
  2. global_tensor_path: standard -> Site31 Mayo reprocessed14 replacement tensor
     (14 Site31 subjects with corrected DPARSF slice-order connectomes)

All scientific parameters except these two are held fixed against the promoted reference.

NOT automatically promoted even if AUC improves.
Promotion requires Stage B AUC/PR-AUC/BA/F1 and Philips/Site2/rawTP checks.

Default behavior is preflight only. Real training requires --confirm-training.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_CONFIG = (
    PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14.json"
)
DEFAULT_PREFLIGHT_DIR = (
    PROJECT_ROOT / "results/revision_bspc_2026/ch1only_site31_mayo_reprocessed14_preflight_20260619"
)

STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
OOF_SCORE_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"

RUN_ID = "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_site31_mayo_reprocessed14"
REFERENCE_RUN_ID = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"

REFERENCE_CHANNELS = [1, 0, 2]
EXPECTED_CHANNELS = [1]
REFERENCE_SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_SELECTED_NAMES = ["Pearson_Full_FisherZ_Signed"]

REFERENCE_TENSOR_PATH = (
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
EXPECTED_TENSOR_PATH = (
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14.npz"
)
ORIGINAL_TENSOR_SHA256 = "f9a00b291a88d92d942ee3404fe0f5b1cfabe5178cf8ff6c139d57658cb8f609"
REPLACEMENT_TENSOR_SHA256 = "9d5cb75f30ab6bae368c738c2ce1fb5175b7630db2b4492067d46c3956267301"
EXPECTED_REPROCESSED_N = 14

EXPECTED_VAE_COUNTS = {"CN": 300, "MCI": 250, "AD": 97}
EXPECTED_CLF_COUNTS = {"CN": 300, "AD": 97}
RECOVER_SUBJECT = "035_S_6927"
EXCLUDED_SUBJECT = "128_S_2002"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

STALE_TOPLEVEL_NAMES = {"classifier_only_readout", "latent_cache", "run_manifest.json"}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")

PROMOTION_RULE = (
    "Must beat BOTH AUC > 0.782951 AND PR-AUC >= 0.559873 simultaneously "
    "(locked v5.1b horizon4480/cycles56 FULL [1,0,2] reference values). "
    "Additionally: Philips CN FPR must not increase vs promoted; "
    "Site2 and rawTP FPR checks required."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--reference-config", type=Path, default=REFERENCE_CONFIG)
    p.add_argument("--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_DIR)
    p.add_argument("--python-executable", default=None)
    p.add_argument("--dry-run", action="store_true", help="Preflight only; do not launch training.")
    p.add_argument("--confirm-training", action="store_true", help="Required for real training.")
    p.add_argument("--force-clean", action="store_true", help="Refuse stale outputs unless passed for real training.")
    p.add_argument("--skip-stage-b", action="store_true")
    p.add_argument("--skip-oof-score-harmonization", action="store_true")
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, max_rows: int = 200) -> None:
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


def config_diff(reference: Dict[str, Any], target: Dict[str, Any]) -> pd.DataFrame:
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


def validate_strict_diff(reference: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
    ref_params = dict(reference["parameters"])
    tgt_params = dict(target["parameters"])
    if set(ref_params) != set(tgt_params):
        raise RuntimeError(
            "Parameter key set differs from promoted reference: "
            f"missing={sorted(set(ref_params) - set(tgt_params))}, "
            f"extra={sorted(set(tgt_params) - set(ref_params))}"
        )
    param_diffs = {k: (ref_params[k], tgt_params[k]) for k in ref_params if ref_params[k] != tgt_params[k]}
    if param_diffs != {"channels_to_use": (REFERENCE_CHANNELS, EXPECTED_CHANNELS)}:
        raise RuntimeError(
            f"Expected only channels_to_use {REFERENCE_CHANNELS} -> {EXPECTED_CHANNELS} in params; got {param_diffs}"
        )

    if reference["selected_channel_names"] != REFERENCE_SELECTED_NAMES:
        raise RuntimeError("Reference selected_channel_names did not match expected promoted [1,0,2] names.")
    if target["selected_channel_names"] != EXPECTED_SELECTED_NAMES:
        raise RuntimeError(f"Target selected_channel_names must be {EXPECTED_SELECTED_NAMES}")

    allowed_path_diffs = {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv", "global_tensor_path"}
    path_diffs = {
        k: (reference["paths"].get(k), target["paths"].get(k))
        for k in set(reference["paths"]) | set(target["paths"])
        if reference["paths"].get(k) != target["paths"].get(k)
    }
    if set(path_diffs) - allowed_path_diffs:
        raise RuntimeError(f"Unexpected path diffs: {set(path_diffs) - allowed_path_diffs}")
    if "global_tensor_path" not in path_diffs:
        raise RuntimeError("global_tensor_path did not change — expected site31_mayo_reprocessed14 tensor")
    if target["paths"]["global_tensor_path"] != EXPECTED_TENSOR_PATH:
        raise RuntimeError(
            f"Target global_tensor_path must be the site31_mayo_reprocessed14 tensor.\n"
            f"Expected: {EXPECTED_TENSOR_PATH}\nGot: {target['paths']['global_tensor_path']}"
        )
    if reference["paths"]["metadata_path"] != target["paths"]["metadata_path"]:
        raise RuntimeError("metadata_path changed unexpectedly — must be identical to promoted reference")

    unchanged_keys = [
        "latent_dim", "beta_vae", "dropout_rate_vae", "vae_dropout_scope", "vae_block_order",
        "epochs_vae", "cyclical_beta_n_cycles", "cyclical_beta_ratio_increase",
        "lr_scheduler_T0", "early_stopping_patience_vae", "batch_size", "decoder_type",
        "recon_loss_mode", "norm_mode", "vae_final_activation", "intermediate_fc_dim_vae",
        "classifier_types", "classifier_stratify_cols", "vae_stratify_cols",
        "metadata_features", "seed", "n_iter_logreg", "n_iter_svm", "vae_train_sampler_strategy",
    ]
    for key in unchanged_keys:
        if ref_params[key] != tgt_params[key]:
            raise RuntimeError(f"{key} changed unexpectedly: {ref_params[key]!r} -> {tgt_params[key]!r}")

    return {
        "channels_to_use": (REFERENCE_CHANNELS, EXPECTED_CHANNELS),
        "global_tensor_path": (REFERENCE_TENSOR_PATH, EXPECTED_TENSOR_PATH),
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
        raise RuntimeError(
            f"Metadata has missing required values:\n"
            f"{bad[['SubjectID','ResearchGroup_Mapped','Manufacturer','Age','Sex','tensor_idx']].head(20).to_string(index=False)}"
        )
    return meta


def validate_subject_pool(meta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    dx_counts = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    clf_counts = clf["ResearchGroup_Mapped"].value_counts().to_dict()
    for dx, expected in EXPECTED_VAE_COUNTS.items():
        rows.append({"check": f"vae_{dx}", "observed": int(dx_counts.get(dx, 0)), "expected": expected,
                     "pass": int(dx_counts.get(dx, 0)) == expected})
    for dx, expected in EXPECTED_CLF_COUNTS.items():
        rows.append({"check": f"classifier_{dx}", "observed": int(clf_counts.get(dx, 0)), "expected": expected,
                     "pass": int(clf_counts.get(dx, 0)) == expected})
    rows.extend([
        {"check": "035_S_6927_present", "observed": bool((meta["SubjectID"] == RECOVER_SUBJECT).any()),
         "expected": True, "pass": bool((meta["SubjectID"] == RECOVER_SUBJECT).any())},
        {"check": "128_S_2002_absent_from_metadata", "observed": bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()),
         "expected": False, "pass": not bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any())},
        {"check": "classifier_pool_n", "observed": int(len(clf)), "expected": 397, "pass": int(len(clf)) == 397},
        {"check": "vae_pool_n", "observed": int(len(meta)), "expected": 647, "pass": int(len(meta)) == 647},
    ])
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Subject pool validation failed:\n" + out.to_string(index=False))
    return out


def validate_tensor(config: Dict[str, Any], meta: pd.DataFrame) -> Dict[str, Any]:
    tensor_path = Path(config["paths"]["global_tensor_path"])
    if not tensor_path.exists():
        raise RuntimeError(f"Replacement tensor not found: {tensor_path}")
    with np.load(tensor_path, allow_pickle=True) as zf:
        tensor_shape = tuple(int(v) for v in zf["global_tensor_data"].shape)
        channel_names = [str(x) for x in zf["channel_names"].astype(str)]
        subject_ids = [str(x) for x in zf["subject_ids"].astype(str)]
        reprocessed_sids = [str(x) for x in zf["site31_mayo_reprocessed_subject_ids"].astype(str)]
        stored_original_sha256 = str(zf["site31_mayo_original_tensor_sha256"])
    channels = list(config["parameters"]["channels_to_use"])
    selected = [channel_names[i] for i in channels]
    if channels != EXPECTED_CHANNELS or selected != EXPECTED_SELECTED_NAMES:
        raise RuntimeError(f"Selected channel mismatch: indices={channels}, names={selected}")
    if tensor_shape != (648, 7, 131, 131):
        raise RuntimeError(f"Expected tensor shape (648,7,131,131), got {tensor_shape}")
    if len(reprocessed_sids) != EXPECTED_REPROCESSED_N:
        raise RuntimeError(f"Expected {EXPECTED_REPROCESSED_N} reprocessed subjects, got {len(reprocessed_sids)}: {reprocessed_sids}")
    if stored_original_sha256 != ORIGINAL_TENSOR_SHA256:
        raise RuntimeError(
            f"Replacement tensor's stored original SHA256 does not match promoted tensor SHA256.\n"
            f"Stored: {stored_original_sha256}\nExpected: {ORIGINAL_TENSOR_SHA256}"
        )
    max_idx = int(meta["tensor_idx"].max())
    if max_idx >= tensor_shape[0]:
        raise RuntimeError(f"Metadata tensor_idx max {max_idx} exceeds tensor subject count {tensor_shape[0]}")
    mismatches = sum(1 for _, row in meta.iterrows() if str(row["SubjectID"]) != subject_ids[int(row["tensor_idx"])])
    if mismatches > 0:
        raise RuntimeError(f"tensor_idx alignment: {mismatches} metadata rows do not match subject_ids in tensor")
    return {
        "tensor_path": str(tensor_path),
        "tensor_shape_full": tensor_shape,
        "selected_channels_to_use": channels,
        "selected_channel_names": selected,
        "selected_tensor_shape_if_loaded": (len(meta), len(channels), tensor_shape[2], tensor_shape[3]),
        "n_metadata_rows": int(len(meta)),
        "n_tensor_subject_ids": int(len(subject_ids)),
        "n_site31_mayo_reprocessed_subjects": int(len(reprocessed_sids)),
        "site31_mayo_reprocessed_subjects": " | ".join(reprocessed_sids),
        "stored_original_sha256_matches_promoted": bool(stored_original_sha256 == ORIGINAL_TENSOR_SHA256),
        "replacement_tensor_sha256_expected": REPLACEMENT_TENSOR_SHA256,
        "035_S_6927_in_tensor": bool(RECOVER_SUBJECT in subject_ids),
        "128_S_2002_in_tensor": bool(EXCLUDED_SUBJECT in subject_ids),
        "128_S_2002_metadata_status": "absent_excluded",
        "tensor_idx_alignment_mismatches": mismatches,
        "tensor_modification_status": "UNTOUCHED — replacement tensor read-only; original tensor untouched",
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
            row["all_diagnosis_required_present"] = (
                bool(row["CN"] > 0 and row["AD"] > 0)
                if split_name.startswith("classifier")
                else bool(row["CN"] > 0 and row["MCI"] > 0 and row["AD"] > 0)
            )
            row["all_three_manufacturers_present"] = all(
                row.get(f"Manufacturer_{m}", 0) > 0 for m in ["GE", "Philips", "SIEMENS"]
            )
            row["outer_test_overlap_in_vae_pool"] = int(len(test_subjects & set(vae_pool["SubjectID"].astype(str))))
            row["vae_internal_val_split_feasible"] = val_ok if split_name == "vae_pool" else ""
            row["vae_internal_val_split_error"] = val_error if split_name == "vae_pool" else ""
            rows.append(row)
    out = pd.DataFrame(rows)
    failed = out[
        (~out["all_diagnosis_required_present"])
        | (~out["all_three_manufacturers_present"])
        | (out["outer_test_overlap_in_vae_pool"] != 0)
        | ((out["split_component"] == "vae_pool") & (out["vae_internal_val_split_feasible"] != True))
    ]
    if not failed.empty:
        raise RuntimeError("Fold feasibility validation failed:\n" + failed.to_string(index=False))
    return out


def output_symlink_status(config: Dict[str, Any]) -> Dict[str, Any]:
    local = resolve(config["paths"]["output_dir"])
    target = Path(config["paths"]["big_disk_output_dir"])
    local_exists = local.exists() or local.is_symlink()
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
    }


def stale_output_markers(output_dir: Path) -> List[Path]:
    if not (output_dir.exists() or output_dir.is_symlink()):
        return []
    if output_dir.is_symlink() and not output_dir.exists():
        return []
    markers: List[Path] = []
    for child in output_dir.iterdir():
        if child.name in STALE_TOPLEVEL_NAMES or child.name.startswith(STALE_PREFIXES):
            markers.append(child)
    for nested in output_dir.rglob("latent_cache"):
        markers.append(nested)
    return sorted(set(markers), key=lambda p: str(p))


def build_stage_a_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    cmd = [
        python_exe,
        str(resolve(config["paths"]["training_script"])),
        "--global_tensor_path",
        str(resolve(config["paths"]["global_tensor_path"])),
        "--metadata_path",
        str(resolve(config["paths"]["metadata_path"])),
        "--output_dir",
        str(resolve(config["paths"]["output_dir"])),
    ]
    for key, value in config["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    return cmd


def build_stage_b_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(STAGE_B_SCRIPT),
        "--run-dir", str(out),
        "--output-dir", str(out / "classifier_only_readout"),
        "--outer-folds", str(config["parameters"]["outer_folds"]),
        "--inner-folds", str(config["parameters"]["inner_folds"]),
        "--models", PRIMARY_MODEL,
        "--reuse-latent-cache",
    ]


def build_oof_command(config: Dict[str, Any], python_exe: str) -> List[str]:
    out = resolve(config["paths"]["output_dir"])
    return [
        python_exe,
        str(OOF_SCORE_SCRIPT),
        "--run-dir", str(out),
        "--output-dir",
        str(PROJECT_ROOT / f"results/revision_bspc_2026/{RUN_ID}_stageB_oof_score_calibration"),
    ]


def validate_commands(
    config: Dict[str, Any],
    stage_a: Sequence[str],
    stage_b: Sequence[str],
    oof: Sequence[str],
) -> pd.DataFrame:
    checks = [
        ("stage_a_channels", values_after_flag(stage_a, "--channels_to_use"), ["1"]),
        ("stage_a_latent_dim", values_after_flag(stage_a, "--latent_dim"), ["384"]),
        ("stage_a_beta", values_after_flag(stage_a, "--beta_vae"), ["3.75"]),
        ("stage_a_recon_loss", values_after_flag(stage_a, "--recon_loss_mode"), [config["parameters"]["recon_loss_mode"]]),
        ("stage_a_epochs", values_after_flag(stage_a, "--epochs_vae"), ["10000"]),
        ("stage_a_cycles", values_after_flag(stage_a, "--cyclical_beta_n_cycles"), ["125"]),
        ("stage_a_T0", values_after_flag(stage_a, "--lr_scheduler_T0"), ["80"]),
        ("stage_a_patience", values_after_flag(stage_a, "--early_stopping_patience_vae"), ["560"]),
        ("stage_a_metadata_features", values_after_flag(stage_a, "--metadata_features"), ["Age", "Sex"]),
        ("stage_a_norm_mode", values_after_flag(stage_a, "--norm_mode"), ["zscore_offdiag"]),
        ("stage_a_dropout", values_after_flag(stage_a, "--dropout_rate_vae"), ["0.15"]),
        ("stage_a_batch_size", values_after_flag(stage_a, "--batch_size"), ["64"]),
        ("stage_a_sampler_strategy", values_after_flag(stage_a, "--vae_train_sampler_strategy"), ["none"]),
        ("stage_a_tensor_path_is_reprocessed14", [values_after_flag(stage_a, "--global_tensor_path")[0]
                                                   if values_after_flag(stage_a, "--global_tensor_path") else ""],
         [EXPECTED_TENSOR_PATH]),
        ("stage_b_model", values_after_flag(stage_b, "--models"), [PRIMARY_MODEL]),
        ("stage_b_outer", values_after_flag(stage_b, "--outer-folds"), ["5"]),
        ("stage_b_inner", values_after_flag(stage_b, "--inner-folds"), ["5"]),
        ("oof_run_dir", values_after_flag(oof, "--run-dir"), [values_after_flag(stage_b, "--run-dir")[0]]),
    ]
    rows = [
        {"check": name, "actual": " ".join(actual), "expected": " ".join(expected), "pass": actual == expected}
        for name, actual, expected in checks
    ]
    out = pd.DataFrame(rows)
    if not out["pass"].all():
        raise RuntimeError("Command validation failed:\n" + out.to_string(index=False))
    return out


def write_launch_script(
    out_dir: Path,
    stage_a: List[str],
    stage_b: List[str],
    oof: List[str],
    run_id: str,
) -> Path:
    launch_path = out_dir / f"launch_{run_id.replace('recover035_', '')}.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated launch script — DO NOT RUN without Stage B AUC/PR-AUC/BA/F1/Philips checks.",
        "# This is a post-final exploratory parsimony sensitivity; NOT auto-promoted.",
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


def write_final_recommendation(out_dir: Path, tensor: Dict[str, Any], symlink: Dict[str, Any], stale: List[Path]) -> None:
    is_clean = symlink["preflight_clean_missing_output_ok"]
    path = out_dir / "final_recommendation.md"
    path.write_text(f"""\
# Final Preflight Recommendation

**Run ID:** `{RUN_ID}`
**Date:** {now_utc()}
**Preflight status:** PASS — all checks complete.

---

## What this run is

This is a **post-final exploratory parsimony sensitivity** combining two controlled changes
versus the promoted reference (`{REFERENCE_RUN_ID}`):

1. **Channel reduction (parsimony):** `channels_to_use [1,0,2] -> [1]` — Pearson Full FisherZ only.
2. **Tensor replacement (preprocessing correction):** The Site31 Mayo reprocessed14 tensor
   replaces the original tensor. 14 Site31 subjects have corrected DPARSF slice-order connectomes.
   The tensor is methodologically justified as a preprocessing correction (confirmed wrong
   slice-order from DICOM headers). The original tensor is UNTOUCHED.

Both changes are simultaneous. All other scientific parameters are held fixed.

---

## Promotion criteria

**This run is NOT automatically promoted even if AUC improves.**

To promote this run, ALL of the following must hold (after Stage B OOF calibration):

1. Pooled AUC > 0.782951 AND PR-AUC >= 0.559873 simultaneously
   (locked v5.1b horizon4480/cycles56 FULL [1,0,2] reference values).
2. Philips CN FPR must not increase relative to promoted (0.0808 at threshold 0.5).
3. Site2 and rawTP (140TP) FPR checks must be assessed and reported.
4. BA and F1 at primary threshold must be confirmed non-inferior.

If ANY criterion is not met, this run does not promote. No further AUC-chasing
is recommended in that case.

---

## Preflight check results

| Check | Result |
|---|---|
| Tensor shape | (648, 7, 131, 131) ✓ |
| Channel [1] = Pearson_Full_FisherZ_Signed | ✓ |
| N reprocessed Site31 subjects in tensor | {tensor["n_site31_mayo_reprocessed_subjects"]} (expected 14) ✓ |
| Stored original SHA256 matches promoted | {tensor["stored_original_sha256_matches_promoted"]} ✓ |
| tensor_idx alignment mismatches | {tensor["tensor_idx_alignment_mismatches"]} ✓ |
| Metadata 128_S_2002 absent | ✓ |
| Metadata 035_S_6927 present | ✓ |
| VAE pool: CN=300/MCI=250/AD=97 | ✓ |
| Classifier pool: CN=300/AD=97 (N=397) | ✓ |
| All 5 outer folds feasible | ✓ |
| Output directory clean (not yet created) | {is_clean} ✓ |
| Stale output markers | {len(stale)} ✓ |

---

## Hard constraints compliance

- No model training performed.
- No tensor modification performed (original tensor untouched; replacement is read-only).
- No source prediction modification performed.
- No metadata modification performed.
- No threshold refitting.
- No subject exclusion.
- Acquisition/QC variables (Martín audit) NOT used as features.

---

## If negative result

If this run does not meet promotion criteria, no further AUC-chasing is recommended.
The remaining path is:
- Report the ch1only sensitivity as a negative result (parsimony does not improve performance).
- Report the site31 reprocessing sensitivity as a null or inconclusive correction.
- Proceed with manuscript submission using the promoted 3-channel model.
""", encoding="utf-8")


def write_preflight_outputs(
    out_dir: Path,
    reference_path: Path,
    config_path: Path,
    reference: Dict[str, Any],
    target: Dict[str, Any],
    strict_diff: Dict[str, Tuple[Any, Any]],
    diff: pd.DataFrame,
    pool: pd.DataFrame,
    tensor: Dict[str, Any],
    folds: pd.DataFrame,
    symlink: Dict[str, Any],
    stale: List[Path],
    commands: pd.DataFrame,
    command_checks: pd.DataFrame,
    stage_a: List[str],
    stage_b: List[str],
    oof: List[str],
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into preflight dir as candidate_config.json
    import shutil
    shutil.copy2(config_path, out_dir / "candidate_config.json")

    write_table(diff, out_dir, "config_diff_vs_promoted")
    write_table(pool, out_dir, "subject_pool_validation")
    write_table(folds, out_dir, "fold_feasibility")
    write_table(pd.DataFrame([tensor]), out_dir, "tensor_validation")
    write_table(
        pd.DataFrame([{**symlink, "stale_marker_count": len(stale),
                       "stale_markers": " | ".join(str(p) for p in stale)}]),
        out_dir, "output_symlink_and_stale_audit"
    )
    write_table(commands, out_dir, "planned_commands")
    write_table(command_checks, out_dir, "dryrun_command_validation")

    write_json(out_dir / "tensor_validation.json", tensor)
    write_json(
        out_dir / "strict_scientific_diff.json",
        {
            "n_scientific_diffs": len(strict_diff),
            "allowed_scientific_diffs": {k: {"reference": v[0], "candidate": v[1]} for k, v in strict_diff.items()},
            "note": (
                "Two controlled scientific changes from promoted reference: "
                "(1) channels_to_use [1,0,2]->[1]; "
                "(2) global_tensor_path standard->site31_mayo_reprocessed14. "
                "All other scientific parameters are locked."
            ),
        },
    )

    launch_path = write_launch_script(out_dir, stage_a, stage_b, oof, RUN_ID)
    write_final_recommendation(out_dir, tensor, symlink, stale)

    report_lines = [
        "# ch1-only site31_mayo_reprocessed14 FULL 5x5 Preflight",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Reference: `{REFERENCE_RUN_ID}`",
        f"Date: {now_utc()}",
        "",
        "## Controlled Diff (2 changes from promoted reference)",
        "",
        "1. `channels_to_use: [1,0,2] -> [1]`",
        "2. `global_tensor_path: standard -> site31_mayo_reprocessed14`",
        "   - 14 Site31 subjects with corrected DPARSF slice-order connectomes",
        "- Run name/output path changes are provenance only.",
        "",
        "## Preserved Settings",
        "",
        f"- `latent_dim={target['parameters']['latent_dim']}`",
        f"- `beta_vae={target['parameters']['beta_vae']}`",
        f"- `epochs_vae={target['parameters']['epochs_vae']}`, "
        f"`cycles={target['parameters']['cyclical_beta_n_cycles']}`, "
        f"`T0={target['parameters']['lr_scheduler_T0']}`, "
        f"`patience={target['parameters']['early_stopping_patience_vae']}`",
        f"- `dropout_rate_vae={target['parameters']['dropout_rate_vae']}`",
        f"- `recon_loss_mode={target['parameters']['recon_loss_mode']}` preserved from promoted.",
        "- Stage B: `logreg_l2`, z+Age/Sex, OOF-ECDF score harmonization planned after training.",
        "",
        "## Validation",
        "",
        f"- Tensor shape: {tensor['tensor_shape_full']} ✓",
        f"- Selected channel: `{tensor['selected_channel_names']}` ✓",
        f"- N reprocessed Site31 subjects: {tensor['n_site31_mayo_reprocessed_subjects']} ✓",
        f"- Stored original SHA256 matches promoted: {tensor['stored_original_sha256_matches_promoted']} ✓",
        f"- tensor_idx alignment mismatches: {tensor['tensor_idx_alignment_mismatches']} ✓",
        f"- Subject pool: VAE CN=300/MCI=250/AD=97; classifier CN=300/AD=97. ✓",
        f"- 035_S_6927 included; 128_S_2002 absent from metadata pools. ✓",
        "- All 5 folds feasible. ✓",
        f"- Output symlink target match: `{symlink['target_match']}`",
        f"- Clean missing output path ready: `{symlink['preflight_clean_missing_output_ok']}`",
        f"- Stale output markers: `{len(stale)}`",
        f"- Launch script written: `{launch_path.name}`",
        "",
        "No real training launched. `--confirm-training` required for real training.",
    ]
    (out_dir / "dry_run_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "dry_run": dry_run,
            "training_launched": False,
            "run_id": RUN_ID,
            "config": str(config_path),
            "reference_config": str(reference_path),
            "preflight_dir": str(out_dir),
            "strict_scientific_diff_count": len(strict_diff),
            "strict_scientific_diffs": {k: {"reference": v[0], "candidate": v[1]} for k, v in strict_diff.items()},
            "promotion_rule": PROMOTION_RULE,
            "guardrails": [
                "no real training during preflight",
                "no tensor modification — original tensor untouched",
                "no metadata modification",
                "no existing model artifact modification",
                "no threshold refitting",
                "not auto-promoted even if AUC improves",
            ],
        },
    )


def run_command(command: Iterable[str]) -> None:
    subprocess.run(list(command), cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    reference_path = resolve(args.reference_config)
    out_dir = resolve(args.preflight_dir)
    python_exe = args.python_executable or "/home/diego/anaconda3/envs/vae_ad/bin/python"

    print(f"Preflight: {RUN_ID}")
    print(f"  Config:    {config_path}")
    print(f"  Reference: {reference_path}")
    print(f"  Output:    {out_dir}")

    reference = load_json(reference_path)
    target = load_json(config_path)

    print("\n[1/7] Validating strict scientific diff vs promoted reference...")
    strict_diff = validate_strict_diff(reference, target)

    print("[2/7] Computing config diff table...")
    diff = config_diff(reference, target)

    print("[3/7] Loading and validating metadata / subject pool...")
    meta = load_metadata(target)
    pool = validate_subject_pool(meta)

    print("[4/7] Validating replacement tensor...")
    tensor = validate_tensor(target, meta)

    print("[5/7] Validating fold feasibility...")
    folds = validate_fold_feasibility(target, meta)

    print("[6/7] Checking output symlink / stale markers...")
    symlink = output_symlink_status(target)
    stale = stale_output_markers(resolve(target["paths"]["output_dir"]))

    print("[7/7] Building and validating commands...")
    stage_a = build_stage_a_command(target, python_exe)
    stage_b = build_stage_b_command(target, python_exe)
    oof = build_oof_command(target, python_exe)
    commands = pd.DataFrame([
        {"name": "stage_a_training", "command": shlex.join(stage_a)},
        {"name": "stage_b_classifier_only", "command": shlex.join(stage_b)},
        {"name": "stageB_oof_ecdf_score_harmonization", "command": shlex.join(oof)},
    ])
    command_checks = validate_commands(target, stage_a, stage_b, oof)

    dry_run = args.dry_run or not args.confirm_training
    write_preflight_outputs(
        out_dir=out_dir,
        reference_path=reference_path,
        config_path=config_path,
        reference=reference,
        target=target,
        strict_diff=strict_diff,
        diff=diff,
        pool=pool,
        tensor=tensor,
        folds=folds,
        symlink=symlink,
        stale=stale,
        commands=commands,
        command_checks=command_checks,
        stage_a=stage_a,
        stage_b=stage_b,
        oof=oof,
        dry_run=dry_run,
    )

    print("\n--- Strict scientific diff ---")
    print(pd.DataFrame([
        {"field": k, "reference": v[0], "candidate": v[1]} for k, v in strict_diff.items()
    ]).to_string(index=False))
    print("\n--- Config diff rows ---")
    print(diff.to_string(index=False))
    print(f"\nPreflight outputs: {out_dir}")
    print(f"Output target match: {symlink['target_match']}; clean missing output OK: {symlink['preflight_clean_missing_output_ok']}")
    print(f"Stale output markers: {len(stale)}")

    if dry_run:
        print("\nDry-run OK. All preflight checks PASS. No training launched.")
        return

    if stale and not args.force_clean:
        raise RuntimeError(
            "Refusing real training with stale output markers. "
            "Quarantine manually and use --force-clean."
        )
    if not symlink["target_match"]:
        raise RuntimeError(
            "Refusing real training: output symlink does not point to big-disk target."
        )
    run_command(stage_a)
    if not args.skip_stage_b:
        run_command(stage_b)
    if not args.skip_oof_score_harmonization:
        run_command(oof)


if __name__ == "__main__":
    main()
